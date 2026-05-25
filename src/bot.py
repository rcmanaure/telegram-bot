"""
Telegram bot — the interface clients see in the demo.

Commands:
  /start    → bienvenida + instrucciones
  /clear    → resetear historial de conversación
  /sources  → mostrar documentos indexados
  /help     → mostrar ayuda
"""
import asyncio
import httpx
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    ApplicationBuilder, CommandHandler, MessageHandler,
    CallbackQueryHandler, ContextTypes, filters,
)
from db import AsyncSessionLocal
from rag import rag_query, get_history
from config import settings

# Map each Telegram user to a namespace
# In production, this would come from DB (user's subscription)
# For demo: everyone uses the default namespace
def get_namespace(user_id: int) -> str:
    return settings.default_namespace


# ─── /start ──────────────────────────────────────────────────────────────────

async def cmd_start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    text = (
        f"👋 ¡Hola {user.first_name}!\n\n"
        "Soy un asistente de documentos. Puedo responder preguntas sobre cualquier "
        "documento que haya sido cargado a mi base de conocimiento.\n\n"
        "Envíame tu pregunta y buscaré la respuesta en los documentos.\n\n"
        "Comandos:\n"
        "• /sources — ver qué documentos conozco\n"
        "• /clear — resetear nuestra conversación\n"
        "• /help — mostrar este mensaje de nuevo"
    )
    await update.message.reply_text(text)


# ─── /help ───────────────────────────────────────────────────────────────────

async def cmd_help(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await cmd_start(update, ctx)


# ─── /sources ────────────────────────────────────────────────────────────────

async def cmd_sources(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    namespace = get_namespace(update.effective_user.id)

    async with httpx.AsyncClient() as client:
        try:
            r = await client.get(f"http://api:{settings.app_port}/stats")
            docs = r.json().get("indexed_documents", [])
            ns_docs = [d for d in docs if d["namespace"] == namespace]

            if not ns_docs:
                await update.message.reply_text(
                    "📭 No hay documentos indexados aún. Cargá un PDF vía la API para comenzar."
                )
                return

            lines = ["📚 *Documentos que conozco:*\n"]
            for doc in ns_docs:
                lines.append(f"• {doc['source']} ({doc['chunks']} fragmentos)")

            await update.message.reply_text(
                "\n".join(lines),
                parse_mode="Markdown"
            )
        except Exception as e:
            print(f"cmd_sources error: {e}")
            await update.message.reply_text("No se pudo obtener la lista de documentos.")


# ─── /clear ──────────────────────────────────────────────────────────────────

async def cmd_clear(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    uid = str(update.effective_user.id)
    namespace = get_namespace(update.effective_user.id)

    async with AsyncSessionLocal() as db:
        from sqlalchemy import text
        await db.execute(
            text("DELETE FROM conversations WHERE telegram_user_id = :uid AND namespace = :ns"),
            {"uid": uid, "ns": namespace}
        )
        await db.commit()

    await update.message.reply_text(
        "🗑️ Conversación borrada. ¡Empezamos de cero!"
    )


# ─── Message handler (main RAG query) ────────────────────────────────────────

async def handle_message(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    question = update.message.text
    uid = str(update.effective_user.id)
    namespace = get_namespace(update.effective_user.id)

    # Show typing indicator
    await ctx.bot.send_chat_action(update.effective_chat.id, "typing")

    try:
        async with AsyncSessionLocal() as db:
            answer, chunks = await rag_query(
                db=db,
                question=question,
                namespace=namespace,
                telegram_user_id=uid,
            )

        # Add source references if we found relevant chunks
        if chunks and chunks[0]["similarity"] > 0.75:
            sources = list({c["source"] for c in chunks[:2]})
            source_note = f"\n\n📎 _Fuentes: {', '.join(sources)}_"
            full_reply = answer + source_note
        else:
            full_reply = answer

        await update.message.reply_text(
            full_reply,
            parse_mode="Markdown",
        )

    except RuntimeError as e:
        await update.message.reply_text(str(e))
    except Exception as e:
        print(f"Error in handle_message: {e}")
        await update.message.reply_text(
            "Lo siento, tuve un problema procesando tu pregunta. Por favor intentá de nuevo."
        )


# ─── App builder ─────────────────────────────────────────────────────────────

def build_bot():
    app = ApplicationBuilder().token(settings.telegram_bot_token).build()

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("sources", cmd_sources))
    app.add_handler(CommandHandler("clear", cmd_clear))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    return app


if __name__ == "__main__":
    bot = build_bot()
    print("🤖 Telegram bot iniciado...")
    bot.run_polling(drop_pending_updates=True)
