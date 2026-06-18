"""
Telegram bot handlers — used by the webhook endpoint in main.py.
Each handler receives tenant context via ctx.bot_data["tenant"].
"""
import base64
import logging

from sqlalchemy import select, func, text
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.error import BadRequest, TelegramError
from telegram.ext import ContextTypes

from config import settings
from config_overlay import get_setting
from db import AsyncSessionLocal, Conversation, DocumentChunk, Tenant, tenant_session
from image_buffer import image_buffer
from limiter import tg_rate_limiter
from llm import is_tool_use_available
from rag import rag_query, _build_source_footer
from services.usage import increment_usage
from services.stt import transcribe_voice
from security import sanitize_user_input

logger = logging.getLogger(__name__)

_TG_MAX_LEN = 4096


async def _send_reply(
    update,
    text: str,
    parse_mode: str = "Markdown",
    reply_markup=None,
) -> None:
    """Send reply_text, splitting into multiple messages when text > 4096 chars.

    Splits on newline boundaries so formatting blocks stay intact.
    The reply_markup (inline keyboard) is only attached to the last message.
    """
    if len(text) <= _TG_MAX_LEN:
        try:
            await update.message.reply_text(text, parse_mode=parse_mode, reply_markup=reply_markup)
        except BadRequest as e:
            if "can't parse entities" in str(e).lower() or "can't find end" in str(e).lower():
                logger.warning("markdown_parse_error — retrying as plain text: %s", e)
                await update.message.reply_text(text, reply_markup=reply_markup)
            else:
                raise
        return

    # Split on newline boundaries into ≤ _TG_MAX_LEN chunks
    parts: list[str] = []
    current_lines: list[str] = []
    current_len = 0
    for line in text.split("\n"):
        line_len = len(line) + 1  # +1 for the newline
        if current_len + line_len > _TG_MAX_LEN and current_lines:
            parts.append("\n".join(current_lines))
            current_lines = [line]
            current_len = line_len
        else:
            current_lines.append(line)
            current_len += line_len
    if current_lines:
        parts.append("\n".join(current_lines))

    for i, part in enumerate(parts):
        markup = reply_markup if i == len(parts) - 1 else None
        try:
            await update.message.reply_text(part, parse_mode=parse_mode, reply_markup=markup)
        except BadRequest as e:
            if "can't parse entities" in str(e).lower() or "can't find end" in str(e).lower():
                logger.warning("markdown_parse_error part=%d — retrying as plain text: %s", i, e)
                await update.message.reply_text(part, reply_markup=markup)
            else:
                raise


def _get_tenant(ctx: ContextTypes.DEFAULT_TYPE) -> Tenant:
    return ctx.bot_data["tenant"]


# ─── /start ──────────────────────────────────────────────────────────────────

async def cmd_start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    tenant = _get_tenant(ctx)

    msg = (
        f"👋 ¡Hola {user.first_name}!\n\n"
        "Soy un asistente de documentos. Puedo responder preguntas sobre cualquier "
        "documento que haya sido cargado a mi base de conocimiento.\n\n"
        "Comandos:\n"
        "• /sources — ver qué documentos conozco\n"
        "• /clear — resetear nuestra conversación\n"
        "• /contactar — contactar con un operador\n"
        "• /help — mostrar este mensaje de nuevo"
    )

    if tenant.example_questions and isinstance(tenant.example_questions, list):
        questions = [q for q in tenant.example_questions if q][:5]
        if questions:
            numbered = "\n".join(f"{i + 1}. {q}" for i, q in enumerate(questions))
            msg += f"\n\nAlgunas preguntas que puedes hacerme:\n{numbered}"

    await update.message.reply_text(msg)


# ─── /help ───────────────────────────────────────────────────────────────────

async def cmd_help(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await cmd_start(update, ctx)


# ─── /contactar ──────────────────────────────────────────────────────────────

async def cmd_contactar(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    tenant = _get_tenant(ctx)
    if tenant.contact_url:
        keyboard = InlineKeyboardMarkup([[
            InlineKeyboardButton("Contactar", url=tenant.contact_url)
        ]])
        await update.message.reply_text(
            "¿Quieres hablar con alguien de nuestro equipo?",
            reply_markup=keyboard,
        )
    else:
        await update.message.reply_text(
            "Para contactarnos escríbenos directamente. ¿En qué te podemos ayudar?"
        )


# ─── /sources ────────────────────────────────────────────────────────────────

async def cmd_sources(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    tenant = _get_tenant(ctx)
    try:
        async with tenant_session(tenant.slug) as db:
            result = await db.execute(
                select(DocumentChunk.source, func.count(DocumentChunk.id).label("chunks"))
                .where(DocumentChunk.namespace == tenant.slug)
                .group_by(DocumentChunk.source)
            )
            rows = result.fetchall()

        if not rows:
            await update.message.reply_text(
                "📭 No hay documentos indexados aún. Carga un PDF vía la API para comenzar."
            )
            return

        lines = ["📚 *Documentos que conozco:*\n"]
        for row in rows:
            name = "Preguntas frecuentes" if row.source == "__faq__" else row.source
            lines.append(f"• {name} ({row.chunks} fragmentos)")
        await update.message.reply_text("\n".join(lines), parse_mode="Markdown")

    except Exception:
        logger.exception("cmd_sources error for tenant %s", tenant.slug)
        await update.message.reply_text("No se pudo obtener la lista de documentos.")


# ─── /clear ──────────────────────────────────────────────────────────────────

async def cmd_clear(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    tenant = _get_tenant(ctx)
    uid = str(update.effective_user.id)

    async with tenant_session(tenant.slug) as db:
        await db.execute(
            text("DELETE FROM conversations WHERE user_id = :uid AND namespace = :ns"),
            {"uid": uid, "ns": tenant.slug},
        )
        await db.commit()

    await update.message.reply_text("🗑️ Conversación borrada. ¡Empezamos de cero!")


# ─── Shared RAG query helper ──────────────────────────────────────────────────

_MAX_PHOTO_BYTES = 5 * 1024 * 1024  # 5 MB


async def _process_question(
    update: Update,
    ctx: ContextTypes.DEFAULT_TYPE,
    question: str,
    reply_suffix: str = "",
    images: list[dict] | None = None,
) -> None:
    tenant = _get_tenant(ctx)
    uid = str(update.effective_user.id)
    rate_key = f"{tenant.slug}:{uid}"

    if tg_rate_limiter.check(rate_key):
        await update.message.reply_text("Demasiados mensajes, espera un minuto.")
        return
    language_code = update.effective_user.language_code

    # UX pre-message: signal multi-source search (only when tool path will actually activate)
    if (
        tenant.web_search_enabled
        and get_setting("web_search_url", "") != ""
        and not images
        and is_tool_use_available()
    ):
        try:
            await ctx.bot.send_message(
                chat_id=update.effective_chat.id,
                text="Procesando tu consulta...",
            )
        except Exception:
            pass  # best-effort — never block the actual response

    try:
        await ctx.bot.send_chat_action(update.effective_chat.id, "typing")
        async with tenant_session(tenant.slug) as db:
            answer, chunks, intent = await rag_query(
                db=db,
                question=question,
                namespace=tenant.slug,
                user_id=uid,
                expertise_area=tenant.expertise_area or "",
                language_code=language_code,
                tenant_id=tenant.id,
                images=images,
                tenant=tenant,
            )
            await increment_usage(db, tenant.id, "queries")

        source_footer = _build_source_footer(chunks, channel="telegram") if intent is None else ""
        full_reply = answer + source_footer

        full_reply += reply_suffix

        reply_markup = None
        # Escalation: add "Contactar" button for off-topic or needs-human intents
        if intent in {"off_topic", "needs_human"} and tenant.contact_url:
            reply_markup = InlineKeyboardMarkup([[
                InlineKeyboardButton("Contactar", url=tenant.contact_url)
            ]])

        await _send_reply(update, full_reply, reply_markup=reply_markup)

    except RuntimeError as e:
        msg = str(e)
        if "LLM service error" in msg or "llm" in msg.lower() or "model" in msg.lower():
            logger.warning("llm_runtime_error uid=%s tenant=%s: %s", uid, tenant.slug, msg)
            await update.message.reply_text(
                "Disculpa, hubo un problema técnico momentáneo. Intenta de nuevo en unos segundos."
            )
        else:
            await update.message.reply_text(msg)
    except Exception:
        logger.exception("handle_message error for tenant %s uid %s", tenant.slug, uid)
        await update.message.reply_text(
            "Lo siento, tuve un problema procesando tu pregunta. Por favor intenta de nuevo."
        )


# ─── Message handler (main RAG query) ────────────────────────────────────────

async def handle_message(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    try:
        text = sanitize_user_input(update.message.text or "")
    except ValueError:
        await update.message.reply_text("Mensaje no permitido.")
        return

    # If the user has pending images in the buffer, flush them now with
    # the text as the question. This handles the common flow:
    #   user sends photo → user types "¿cuánto cuesta esto?"
    uid = str(update.effective_user.id)
    buf_prefix = f"tg:{uid}:"
    flushed = await image_buffer.flush_by_prefix(buf_prefix, override_question=text)
    if flushed:
        # The flushed images will be processed via on_flush → _process_question.
        # The text message itself is the question for those images — skip
        # processing it again as a standalone query.
        return

    await _process_question(update, ctx, text)


# ─── Voice note handler ───────────────────────────────────────────────────────

async def handle_voice(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    try:
        voice = update.message.voice
        from config_overlay import get_setting
        if not get_setting("groq_api_key", settings.groq_api_key):
            await update.message.reply_text("Las notas de voz no están habilitadas aún.")
            return
        MAX_VOICE_BYTES = 10 * 1024 * 1024
        if voice.file_size and voice.file_size > MAX_VOICE_BYTES:
            await update.message.reply_text("La nota de voz es demasiado larga.")
            return
        await ctx.bot.send_chat_action(update.effective_chat.id, "typing")
        try:
            tg_file = await ctx.bot.get_file(voice.file_id)
            audio: bytes = bytes(await tg_file.download_as_bytearray())
        except TelegramError:
            await update.message.reply_text("No pude descargar la nota de voz.")
            return
        try:
            transcript = await transcribe_voice(audio)
        except RuntimeError as e:
            await update.message.reply_text(str(e))
            return
        if not transcript.strip():
            await update.message.reply_text("No pude entender la nota de voz.")
            return
        try:
            transcript = sanitize_user_input(transcript)
        except ValueError:
            await update.message.reply_text("Mensaje no permitido.")
            return
        echo = f"\n\n_🎤 Escuché: «{transcript[:120]}{'...' if len(transcript) > 120 else ''}»_"

        # If the user has pending images in the buffer, flush them with
        # the voice transcript as the question
        uid = str(update.effective_user.id)
        buf_prefix = f"tg:{uid}:"
        flushed = await image_buffer.flush_by_prefix(buf_prefix, override_question=transcript)
        if flushed:
            # Images will be processed with the voice transcript as question
            return

        await _process_question(update, ctx, transcript, reply_suffix=echo)
    except Exception:
        logger.exception(
            "handle_voice error file_id=%s",
            update.message.voice.file_id if update.message.voice else "unknown",
        )
        await update.message.reply_text("Lo siento, tuve un problema. Intenta de nuevo.")


# ─── Photo handler (with multi-image buffer) ────────────────────────────────

async def handle_photo(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    photos = update.message.photo
    if not photos:
        return

    # Telegram sends multiple sizes; use the largest
    photo = max(photos, key=lambda p: p.file_size or 0)

    if photo.file_size and photo.file_size > _MAX_PHOTO_BYTES:
        await update.message.reply_text("La imagen es demasiado grande (máx 5 MB).")
        return

    # Send typing indicator immediately (before buffer + download)
    await ctx.bot.send_chat_action(update.effective_chat.id, "typing")

    try:
        tg_file = await ctx.bot.get_file(photo.file_id)
        image_bytes: bytes = bytes(await tg_file.download_as_bytearray())
    except TelegramError:
        await update.message.reply_text("No pude descargar la imagen. Intenta de nuevo.")
        return

    image_b64 = base64.b64encode(image_bytes).decode("utf-8")

    # Sniff MIME type; default to jpeg for unknown formats
    try:
        import filetype as _ft
        kind = _ft.guess(image_bytes)
        mime = kind.mime if kind else "image/jpeg"
    except Exception:
        mime = "image/jpeg"

    # Caption as question (empty string if no caption — buffer provides default)
    question = update.message.caption or ""
    if question.strip():
        try:
            question = sanitize_user_input(question)
        except ValueError:
            await update.message.reply_text("Mensaje no permitido.")
            return

    # Determine buffer key: group by media_group_id (album) or user-level pending
    uid = str(update.effective_user.id)
    media_group_id = getattr(update.message, "media_group_id", None)
    if media_group_id:
        buf_key = f"tg:{uid}:{media_group_id}"
    else:
        buf_key = f"tg:{uid}:_pending_"

    # Flush callback: process collected images through RAG
    async def on_flush(images_dicts: list[dict], combined_question: str) -> None:
        await _process_question(update, ctx, combined_question, images=images_dicts)

    error = image_buffer.add_image(
        key=buf_key,
        b64=image_b64,
        mime=mime,
        question=question,
        on_flush=on_flush,
    )
    if error:
        await update.message.reply_text(error)

    logger.info(
        "handle_photo user=%s bytes=%d mime=%s group_id=%s buf_key=%s",
        uid, len(image_bytes), mime, media_group_id, buf_key,
    )


# ─── Handler registration ──────────────────────────────────────────────────────

def register_handlers(tg_app):
    """Register all Telegram bot command and message handlers."""
    from telegram.ext import CommandHandler, MessageHandler, filters
    tg_app.add_handler(CommandHandler("start", cmd_start))
    tg_app.add_handler(CommandHandler("help", cmd_help))
    tg_app.add_handler(CommandHandler("sources", cmd_sources))
    tg_app.add_handler(CommandHandler("clear", cmd_clear))
    tg_app.add_handler(CommandHandler("contactar", cmd_contactar))
    tg_app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    tg_app.add_handler(MessageHandler(filters.VOICE, handle_voice))
    tg_app.add_handler(MessageHandler(filters.PHOTO, handle_photo))