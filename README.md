# 🤖 FAQ Bot with RAG — Telegram + FastAPI + pgvector

A production-ready AI chatbot that answers questions from your business documents.
Upload PDFs → bot answers questions citing the exact source.

Built as a portfolio demo by **Ruben C.** — AI Engineer & Backend Developer

## 🎥 Demo flow
1. Upload a PDF (product catalog, FAQ doc, policy, manual)
2. Ask the Telegram bot anything about it
3. Bot answers in natural language, citing the exact section

## 🏗️ Architecture

```
User (Telegram)
      ↓
Telegram Bot (python-telegram-bot)
      ↓
FastAPI Backend
      ↓
┌─────────────────────────────────┐
│  RAG Pipeline                   │
│  1. Embed query (OpenRouter)    │
│  2. pgvector similarity search  │
│  3. Claude Haiku answers with   │
│     retrieved context           │
└─────────────────────────────────┘
      ↓
PostgreSQL + pgvector
```

## 🔧 Tech Stack

- **Backend:** Python 3.11, FastAPI, SQLAlchemy async
- **Bot:** python-telegram-bot 21.x
- **AI:** OpenRouter — `anthropic/claude-haiku-4.5` for answers, `openai/text-embedding-3-small` for embeddings
- **Vector DB:** PostgreSQL + pgvector extension
- **Containerization:** Docker + docker-compose

> All AI calls (LLM + embeddings) go through a single `OPENROUTER_API_KEY`. No separate OpenAI key needed.

## 🚀 Quick Start

```bash
# 1. Clone and configure
cp .env.example .env
# Fill in OPENROUTER_API_KEY and TELEGRAM_BOT_TOKEN in .env

# 2. Start all services (postgres → api → bot, in health-checked order)
docker compose up -d

# 3. Seed with demo data (Acme Gym FAQ)
docker compose exec api python scripts/seed_demo.py

# 4. Upload your own document
curl -X POST http://localhost:8000/upload \
  -F "file=@your_document.pdf" \
  -F "namespace=my_company"

# 5. Start chatting on Telegram
# Find your bot and send /start
```

## 📁 Project Structure

```
├── src/
│   ├── main.py          # FastAPI app — /upload, /health, /stats, /namespace
│   ├── bot.py           # Telegram bot handlers
│   ├── rag.py           # RAG pipeline (embed, search, answer)
│   ├── db.py            # Database models + pgvector setup
│   └── config.py        # Settings from .env
├── documents/           # Place PDFs here for manual upload
├── scripts/
│   └── seed_demo.py     # Seeds DB with Acme Gym sample data
├── tests/
│   └── test_rag_pipeline.py  # Unit + integration smoke tests
├── docker-compose.yml
├── Dockerfile
└── requirements.txt
```

## 🔑 Environment Variables

| Variable | Required | Description |
|---|---|---|
| `OPENROUTER_API_KEY` | Yes | OpenRouter key — covers both LLM and embeddings |
| `TELEGRAM_BOT_TOKEN` | Yes | From [@BotFather](https://t.me/BotFather) |
| `DATABASE_URL` | Yes | Use `@postgres:5432` inside Docker, `@localhost:5432` for local dev |
| `LLM_MODEL` | No | Default: `anthropic/claude-haiku-4.5` |
| `EMBEDDING_MODEL` | No | Default: `openai/text-embedding-3-small` (via OpenRouter) |

## 🧪 Tests

```bash
# Unit tests (no external services needed)
docker compose exec api python -m pytest tests/ -v -k "not integration"

# Integration tests (requires running services + valid API keys)
docker compose exec api python -m pytest tests/ -v
```

## 🤖 Bot Commands

| Command | Description |
|---|---|
| `/start` | Welcome message + instructions |
| `/sources` | List indexed documents |
| `/clear` | Reset conversation history |
| `/help` | Show help |

## 💼 Use Cases (for clients)

- Customer support bot trained on your product docs
- Internal HR bot trained on company policies
- Real estate bot trained on property listings
- Legal bot trained on contracts / terms

## 📊 Performance

- Embedding: ~200ms per query
- Vector search: ~50ms (pgvector cosine similarity)
- Total response time: ~1.5-2.5s end-to-end
