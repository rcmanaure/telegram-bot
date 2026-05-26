"""
Telegram bot handlers — used by the webhook endpoint in main.py.
Each handler receives tenant context via ctx.bot_data["tenant"].
"""
import logging
from sqlalchemy import select, func, text
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import ContextTypes

from db import AsyncSessionLocal, Conversation, DocumentChunk, Tenant
from rag import rag_query

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
            lines.append(f"• {row.source} ({row.chunks} fragmentos)")
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


# ─── Message handler (main RAG query) ────────────────────────────────────────

async def handle_message(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    tenant = _get_tenant(ctx)
    question = update.message.text
    uid = str(update.effective_user.id)
    language_code = update.effective_user.language_code

    await ctx.bot.send_chat_action(update.effective_chat.id, "typing")

    try:
        async with AsyncSessionLocal() as db:
            answer, chunks, intent = await rag_query(
                db=db,
                question=question,
                namespace=tenant.slug,
                user_id=uid,
                expertise_area=tenant.expertise_area or "",
                language_code=language_code,
                tenant_id=tenant.id,
            )

        if chunks and chunks[0]["similarity"] > 0.75:
            sources = list({c["source"] for c in chunks[:2]})
            full_reply = answer + f"\n\n📎 _Fuentes: {', '.join(sources)}_"
        else:
            full_reply = answer

        # Answered from docs — append follow-up prompt (not saved to DB history)
        if intent is None:
            full_reply += "\n\n¿Hay algo más en lo que pueda ayudarte?"

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
