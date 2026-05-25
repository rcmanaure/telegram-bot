"""
Telegram bot — the interface clients see in the demo.

Commands:
  /start    → welcome + instructions
  /clear    → reset conversation history
  /sources  → show indexed documents
  /help     → show help

Any other message → RAG query
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
        f"👋 Hi {user.first_name}!\n\n"
        "I'm a document assistant. I can answer questions about any document "
        "that has been uploaded to my knowledge base.\n\n"
        "Just send me your question and I'll search the documents for the answer.\n\n"
        "Commands:\n"
        "• /sources — see what documents I know about\n"
        "• /clear — reset our conversation\n"
        "• /help — show this message again"
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
                    "📭 No documents indexed yet. Upload a PDF via the API to get started."
                )
                return

            lines = ["📚 *Documents I know about:*\n"]
            for doc in ns_docs:
                lines.append(f"• {doc['source']} ({doc['chunks']} chunks)")

            await update.message.reply_text(
                "\n".join(lines),
                parse_mode="Markdown"
            )
        except Exception as e:
            print(f"cmd_sources error: {e}")
            await update.message.reply_text("Could not fetch document list.")


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
        "🗑️ Conversation cleared. Fresh start!"
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
            source_note = f"\n\n📎 _Sources: {', '.join(sources)}_"
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
            "Sorry, I ran into an issue processing your question. Please try again."
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
    print("🤖 Telegram bot started...")
    bot.run_polling(drop_pending_updates=True)
