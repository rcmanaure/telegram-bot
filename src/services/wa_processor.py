"""WhatsApp message processing — background task handler."""
import asyncio
import logging

from config_overlay import get_setting
from db import AsyncSessionLocal, Tenant, tenant_session
from channels.whatsapp import WhatsAppAdapter, check_wa_service_window, update_wa_service_window, send_wa_template
from channels.protocol import ChannelButton, ChannelSendError
from image_buffer import image_buffer
from limiter import wa_rate_limiter
from llm import is_tool_use_available
from rag import rag_query, _log_unanswered, _build_source_footer
from services.usage import increment_usage
from security import sanitize_user_input

logger = logging.getLogger(__name__)


async def _send_wa_reply(
    adapter: WhatsAppAdapter,
    user_id: str,
    answer: str,
    chunks: list[dict] | None = None,
    intent: str | None = None,
    tenant: Tenant | None = None,
) -> None:
    """Send a WhatsApp reply with sources footer and escalation button.

    Shared by _wa_process_flushed and handle_wa_message to avoid
    divergent reply formatting.
    """
    source_footer = _build_source_footer(chunks, channel="whatsapp") if intent is None else ""

    # Escalation: add "Contactar" button for off-topic or needs-human intents
    buttons = None
    if intent in {"off_topic", "needs_human"} and tenant and tenant.contact_url:
        buttons = [ChannelButton(label="Contactar", url=tenant.contact_url)]

    try:
        await adapter.send_reply(user_id, answer + source_footer, buttons=buttons)
    except ChannelSendError as e:
        logger.warning("wa_send_failed user=%s: %s — retrying once", user_id, e)
        await asyncio.sleep(2)
        try:
            await adapter.send_reply(user_id, answer + source_footer)
        except ChannelSendError:
            logger.error("wa_send_failed_retry user=%s — giving up", user_id)


def create_wa_adapter(tenant: Tenant) -> WhatsAppAdapter | None:
    """Create a WhatsApp adapter for the tenant, or None if WA not configured."""
    if not tenant.wa_phone_number_id or not tenant.wa_access_token:
        return None
    if "whatsapp" not in (tenant.channels or "telegram"):
        return None
    return WhatsAppAdapter(
        phone_number_id=tenant.wa_phone_number_id,
        access_token=tenant.wa_access_token,
        app_secret=tenant.wa_app_secret or "",
        verify_token=tenant.wa_verify_token or "",
        business_id=tenant.wa_business_id,
    )


async def _wa_process_flushed(
    tenant: Tenant,
    user_id: str,
    namespace: str,
    images: list[dict],
    question: str,
) -> None:
    """Process flushed images from the image buffer (WA channel).

    Creates its own adapter and DB session since this runs after
    the original handle_wa_message context has exited.
    """
    adapter = create_wa_adapter(tenant)
    if not adapter:
        logger.warning("wa_flush_no_adapter tenant=%s", tenant.slug)
        return

    async with adapter:
        # Send typing indicator before RAG query
        try:
            await adapter.send_chat_action(user_id)
        except Exception:
            pass

        async with tenant_session(namespace) as db:
            try:
                answer, chunks, intent = await rag_query(
                    db=db,
                    question=question,
                    namespace=namespace,
                    user_id=user_id,
                    expertise_area=tenant.expertise_area or "",
                    tenant_id=tenant.id,
                    channel="whatsapp",
                    images=images,
                    tenant=tenant,
                )
                await increment_usage(db, tenant.id, "queries")
            except Exception as e:
                logger.error("wa_rag_error user=%s: %s", user_id, e)
                try:
                    await adapter.send_reply(user_id, "Lo siento, hubo un error. Intenta de nuevo en un momento.")
                except ChannelSendError:
                    pass
                return

        await _send_wa_reply(adapter, user_id, answer, chunks=chunks, intent=intent, tenant=tenant)


async def handle_wa_message(
    tenant: Tenant,
    wa_msg: "ChannelMessage",
) -> None:
    """Background task: process a single WA message through the RAG pipeline.

    Creates its own DB session since the request-scoped session from the
    webhook handler will be closed by the time this task runs.
    """
    user_id = wa_msg.user_id
    namespace = tenant.slug

    # Create adapter once for the entire message lifecycle
    adapter = create_wa_adapter(tenant)
    if not adapter:
        logger.warning("wa_no_adapter tenant=%s", tenant.slug)
        return

    async with adapter:
        # Unsupported media (video, sticker, document) — not image or voice
        if wa_msg.text is None and wa_msg.media_type not in ("voice", "image"):
            try:
                await adapter.send_reply(
                    user_id,
                    "Solo puedo procesar texto, imágenes y notas de voz. "
                    "Escribe tu consulta o envía una imagen o nota de voz.",
                )
            except ChannelSendError:
                logger.warning("wa_send_failed user=%s — media fallback", user_id)
            return

        # Voice note — placeholder (not yet implemented for WA)
        if wa_msg.media_type == "voice":
            try:
                await adapter.send_reply(
                    user_id,
                    "Las notas de voz aún no están disponibles. Escribe tu consulta por texto.",
                )
            except ChannelSendError:
                logger.warning("wa_send_failed user=%s — voice fallback", user_id)
            return

        # Image message — download and base64-encode for vision model
        image_b64: str | None = None
        image_mime: str = "image/jpeg"
        if wa_msg.media_type == "image" and wa_msg.media_url:
            try:
                import base64 as _b64
                image_bytes = await adapter.download_media(wa_msg.media_url)
                if len(image_bytes) > 5 * 1024 * 1024:
                    try:
                        await adapter.send_reply(user_id, "La imagen es demasiado grande (máx 5 MB).")
                    except ChannelSendError:
                        pass
                    return
                image_b64 = _b64.b64encode(image_bytes).decode("utf-8")
                try:
                    import filetype as _ft
                    kind = _ft.guess(image_bytes)
                    image_mime = kind.mime if kind else "image/jpeg"
                except Exception:
                    pass
            except Exception as e:
                logger.warning("wa_image_download_failed user=%s: %s", user_id, e)
                try:
                    await adapter.send_reply(user_id, "No pude descargar la imagen. Intenta de nuevo.")
                except ChannelSendError:
                    pass
                return

        # Rate limit (20/60s per tenant:user, independent of TG)
        # Check BEFORE sanitize so injection probes don't bypass rate limiting
        rate_key = f"{namespace}:{user_id}"
        if wa_rate_limiter.check(rate_key):
            logger.warning("wa_rate_limit key=%s", rate_key)
            return

        # Text handling: caption or user text
        text = wa_msg.text or ""
        if not image_b64 and not text.strip():
            return
        if text.strip():
            try:
                text = sanitize_user_input(text)
            except ValueError:
                try:
                    await adapter.send_reply(user_id, "Tu mensaje contiene contenido no permitido.")
                except ChannelSendError:
                    pass
                return

        # Service window check (before buffering)
        async with tenant_session(namespace) as db:
            within_window = await check_wa_service_window(db, tenant.id, user_id)
            if not within_window:
                if tenant.wa_reengagement_template:
                    try:
                        await send_wa_template(adapter, user_id, tenant.wa_reengagement_template)
                    except ChannelSendError:
                        logger.warning("wa_template_failed user=%s template=%s", user_id, tenant.wa_reengagement_template)
                else:
                    logger.warning("wa_outside_window_no_template user=%s — logging unanswered", user_id)
                    await _log_unanswered(db, namespace, text, user_id, "needs_human", tenant.id)
                return
            await update_wa_service_window(db, tenant.id, user_id)

        # Image: add to buffer with flush callback
        # The flush callback creates its own adapter since this one closes on exit
        if image_b64:
            buf_key = f"wa:{namespace}:{user_id}"

            async def on_flush(images_dicts: list[dict], combined_question: str) -> None:
                await _wa_process_flushed(tenant, user_id, namespace, images_dicts, combined_question)

            question = text or ""
            error = image_buffer.add_image(
                key=buf_key,
                b64=image_b64,
                mime=image_mime,
                question=question,
                on_flush=on_flush,
            )
            if error:
                try:
                    await adapter.send_reply(user_id, error)
                except ChannelSendError:
                    pass
            return

        # Text-only (no image): check if images are pending in buffer first
        buf_key_prefix = f"wa:{namespace}:{user_id}"
        flushed = await image_buffer.flush_by_prefix(buf_key_prefix, override_question=text)
        if flushed:
            # Images will be processed with this text as the question via on_flush
            return

        # No pending images — process text immediately

        # UX pre-message: signal multi-source search (only when tool path will activate)
        if (
            tenant.web_search_enabled
            and get_setting("web_search_url", "") != ""
            and is_tool_use_available()
        ):
            try:
                await adapter.send_reply(user_id, "Procesando tu consulta...")
            except Exception:
                pass  # best-effort

        async with tenant_session(namespace) as db:
            try:
                answer, chunks, intent = await rag_query(
                    db=db,
                    question=text,
                    namespace=namespace,
                    user_id=user_id,
                    expertise_area=tenant.expertise_area or "",
                    tenant_id=tenant.id,
                    channel="whatsapp",
                    tenant=tenant,
                )
                await increment_usage(db, tenant.id, "queries")
            except Exception as e:
                logger.error("wa_rag_error user=%s: %s", user_id, e)
                try:
                    await adapter.send_reply(user_id, "Lo siento, hubo un error. Intenta de nuevo en un momento.")
                except ChannelSendError:
                    pass
                return

        await _send_wa_reply(adapter, user_id, answer, chunks=chunks, intent=intent, tenant=tenant)