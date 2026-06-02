"""WhatsApp message processing — background task handler."""
import asyncio
import logging

from db import AsyncSessionLocal, Tenant
from channels.whatsapp import WhatsAppAdapter, check_wa_service_window, update_wa_service_window, send_wa_template
from channels.protocol import ChannelButton, ChannelSendError
from limiter import wa_rate_limiter
from rag import rag_query, _log_unanswered
from security import sanitize_user_input

logger = logging.getLogger(__name__)


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

    # Handle feedback button callbacks (thumbs up/down)
    if wa_msg.reply_to and wa_msg.reply_to.startswith("fb:"):
        from db import Feedback
        parts = wa_msg.reply_to.split(":")
        if len(parts) >= 3:
            rating = "positive" if parts[1] == "pos" else "negative"
            try:
                async with AsyncSessionLocal() as db:
                    db.add(Feedback(
                        tenant_id=tenant.id,
                        user_id=user_id,
                        namespace=namespace,
                        rating=rating,
                    ))
                    await db.commit()
            except Exception:
                logger.exception("wa_feedback_store_failed user=%s", user_id)
            # Acknowledge feedback
            adapter = create_wa_adapter(tenant)
            if adapter:
                emoji = "👍" if rating == "positive" else "👎"
                try:
                    async with adapter:
                        await adapter.send_reply(user_id, f"{emoji} ¡Gracias por tu feedback!")
                except ChannelSendError:
                    pass
        return

    # Create adapter once for the entire message lifecycle
    adapter = create_wa_adapter(tenant)
    if not adapter:
        logger.warning("wa_no_adapter tenant=%s", tenant.slug)
        return

    image_b64: str | None = None
    image_mime: str = "image/jpeg"

    async with adapter:
        # Unsupported media (video, sticker, document) — not image or voice
        if wa_msg.text is None and wa_msg.media_type not in ("voice", "image"):
            try:
                await adapter.send_reply(
                    user_id,
                    "Solo puedo procesar texto, imágenes y notas de voz. "
                    "Escribí tu consulta o enviá una imagen o nota de voz.",
                )
            except ChannelSendError:
                logger.warning("wa_send_failed user=%s — media fallback", user_id)
            return

        # Voice note — placeholder (not yet implemented for WA)
        if wa_msg.media_type == "voice":
            try:
                await adapter.send_reply(
                    user_id,
                    "Las notas de voz aún no están disponibles. Escribí tu consulta por texto.",
                )
            except ChannelSendError:
                logger.warning("wa_send_failed user=%s — voice fallback", user_id)
            return

        # Image message — download and base64-encode for vision model
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
                    await adapter.send_reply(user_id, "No pude descargar la imagen. Intentá de nuevo.")
                except ChannelSendError:
                    pass
                return

        # Rate limit (20/60s per tenant:user, independent of TG)
        # Check BEFORE sanitize so injection probes don't bypass rate limiting
        rate_key = f"{namespace}:{user_id}"
        if wa_rate_limiter.check(rate_key):
            logger.warning("wa_rate_limit key=%s", rate_key)
            return

        # Text or image-with-caption message — process through RAG
        text = wa_msg.text or ("¿Qué querés saber sobre esta imagen?" if image_b64 else "")
        if not text.strip():
            return

        try:
            text = sanitize_user_input(text)
        except ValueError:
            try:
                await adapter.send_reply(user_id, "Tu mensaje contiene contenido no permitido.")
            except ChannelSendError:
                pass
            return

        # DB operations — fresh session for background task
        async with AsyncSessionLocal() as db:
            # 24h service window check
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

            # Update service window timestamp
            await update_wa_service_window(db, tenant.id, user_id)

            # RAG query
            try:
                answer, chunks, intent = await rag_query(
                    db=db,
                    question=text,
                    namespace=namespace,
                    user_id=user_id,
                    expertise_area=tenant.expertise_area or "",
                    tenant_id=tenant.id,
                    channel="whatsapp",
                    image_b64=image_b64,
                    image_mime=image_mime,
                    tenant=tenant,
                )
            except Exception as e:
                logger.error("wa_rag_error user=%s: %s", user_id, e)
                try:
                    await adapter.send_reply(user_id, "Lo siento, hubo un error. Intentá de nuevo en un momento.")
                except ChannelSendError:
                    pass
                return

            # Send reply with sources footer + feedback buttons
            source_footer = ""
            if chunks:
                sources = set(c["source"] for c in chunks if c.get("source"))
                if sources:
                    source_footer = "\n\n📎 Fuentes: " + ", ".join(sources)

            # Add thumbs up/down feedback buttons (WhatsApp allows max 3)
            buttons = [
                ChannelButton(label="👍", callback_data=f"fb:pos:{namespace}"),
                ChannelButton(label="👎", callback_data=f"fb:neg:{namespace}"),
            ]
            # Escalation: replace feedback buttons with contact button
            if intent in {"off_topic", "needs_human"} and tenant.contact_url:
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