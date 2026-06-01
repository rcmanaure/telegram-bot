"""
Telegram bot handlers — used by the webhook endpoint in main.py.
Each handler receives tenant context via ctx.bot_data["tenant"].
"""
import base64
import logging

from sqlalchemy import select, func, text
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.error import TelegramError
from telegram.ext import ContextTypes

from config import settings
from db import AsyncSessionLocal, Conversation, DocumentChunk, Tenant
from limiter import tg_rate_limiter
from rag import rag_query
from services.stt import transcribe_voice
from security import sanitize_user_input

logger = logging.getLogger(__name__)

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
            msg += f"\n\nAlgunas preguntas que podés hacerme:\n{numbered}"

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
            "¿Querés hablar con alguien de nuestro equipo?",
            reply_markup=keyboard,
        )
    else:
        await update.message.reply_text(
            "Para contactarnos escribinos directamente. ¿En qué te podemos ayudar?"
        )


# ─── /sources ────────────────────────────────────────────────────────────────

async def cmd_sources(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    tenant = _get_tenant(ctx)
    try:
        async with AsyncSessionLocal() as db:
            result = await db.execute(
                select(DocumentChunk.source, func.count(DocumentChunk.id).label("chunks"))
                .where(DocumentChunk.namespace == tenant.slug)
                .group_by(DocumentChunk.source)
            )
            rows = result.fetchall()

        if not rows:
            await update.message.reply_text(
                "📭 No hay documentos indexados aún. Cargá un PDF vía la API para comenzar."
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

    async with AsyncSessionLocal() as db:
        await db.execute(
            text("DELETE FROM conversations WHERE user_id = :uid AND namespace = :ns"),
            {"uid": uid, "ns": tenant.slug},
        )
        await db.commit()

    await update.message.reply_text("🗑️ Conversación borrada. ¡Empezamos de cero!")


# ─── Shared RAG query helper ──────────────────────────────────────────────────

_PHOTO_DEFAULT_QUESTION = "¿Qué querés saber sobre esta imagen?"
_MAX_PHOTO_BYTES = 5 * 1024 * 1024  # 5 MB


async def _process_question(
    update: Update,
    ctx: ContextTypes.DEFAULT_TYPE,
    question: str,
    reply_suffix: str = "",
    image_b64: str | None = None,
    image_mime: str = "image/jpeg",
) -> None:
    tenant = _get_tenant(ctx)
    uid = str(update.effective_user.id)
    rate_key = f"{tenant.slug}:{uid}"

    if tg_rate_limiter.check(rate_key):
        await update.message.reply_text("Demasiados mensajes, esperá un minuto.")
        return
    language_code = update.effective_user.language_code

    try:
        await ctx.bot.send_chat_action(update.effective_chat.id, "typing")
        async with AsyncSessionLocal() as db:
            answer, chunks, intent = await rag_query(
                db=db,
                question=question,
                namespace=tenant.slug,
                user_id=uid,
                expertise_area=tenant.expertise_area or "",
                language_code=language_code,
                tenant_id=tenant.id,
                image_b64=image_b64,
                image_mime=image_mime,
            )

        if chunks and chunks[0]["similarity"] > 0.75:
            sources = list({c["source"] for c in chunks[:2]})
            full_reply = answer + f"\n\n📎 _Fuentes: {', '.join(sources)}_"
        else:
            full_reply = answer

        full_reply += reply_suffix

        reply_markup = None
        if intent in {"off_topic", "needs_human"} and tenant.contact_url:
            reply_markup = InlineKeyboardMarkup([[
                InlineKeyboardButton("Contactar", url=tenant.contact_url)
            ]])

        await update.message.reply_text(
            full_reply, parse_mode="Markdown", reply_markup=reply_markup
        )

    except RuntimeError as e:
        await update.message.reply_text(str(e))
    except Exception:
        logger.exception("handle_message error for tenant %s uid %s", tenant.slug, uid)
        await update.message.reply_text(
            "Lo siento, tuve un problema procesando tu pregunta. Por favor intentá de nuevo."
        )


# ─── Message handler (main RAG query) ────────────────────────────────────────

async def handle_message(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    try:
        text = sanitize_user_input(update.message.text or "")
    except ValueError:
        await update.message.reply_text("Mensaje no permitido.")
        return
    await _process_question(update, ctx, text)


# ─── Voice note handler ───────────────────────────────────────────────────────

async def handle_voice(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    try:
        voice = update.message.voice
        if not settings.groq_api_key:
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
        await _process_question(update, ctx, transcript, reply_suffix=echo)
    except Exception:
        logger.exception(
            "handle_voice error file_id=%s",
            update.message.voice.file_id if update.message.voice else "unknown",
        )
        await update.message.reply_text("Lo siento, tuve un problema. Intentá de nuevo.")


# ─── Photo handler ────────────────────────────────────────────────────────────

async def handle_photo(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    photos = update.message.photo
    if not photos:
        return

    # Telegram sends multiple sizes; use the largest
    photo = max(photos, key=lambda p: p.file_size or 0)

    if photo.file_size and photo.file_size > _MAX_PHOTO_BYTES:
        await update.message.reply_text("La imagen es demasiado grande (máx 5 MB).")
        return

    await ctx.bot.send_chat_action(update.effective_chat.id, "typing")

    try:
        tg_file = await ctx.bot.get_file(photo.file_id)
        image_bytes: bytes = bytes(await tg_file.download_as_bytearray())
    except TelegramError:
        await update.message.reply_text("No pude descargar la imagen. Intentá de nuevo.")
        return

    image_b64 = base64.b64encode(image_bytes).decode("utf-8")

    # Sniff MIME type; default to jpeg for unknown formats
    try:
        import filetype as _ft
        kind = _ft.guess(image_bytes)
        mime = kind.mime if kind else "image/jpeg"
    except Exception:
        mime = "image/jpeg"

    question = update.message.caption or _PHOTO_DEFAULT_QUESTION
    try:
        question = sanitize_user_input(question)
    except ValueError:
        await update.message.reply_text("Mensaje no permitido.")
        return

    logger.info("handle_photo user=%s bytes=%d mime=%s", update.effective_user.id, len(image_bytes), mime)
    await _process_question(update, ctx, question, image_b64=image_b64, image_mime=mime)


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
