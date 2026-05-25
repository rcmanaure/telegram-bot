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
│  1. Embed query (OpenAI)        │
│  2. pgvector similarity search  │
│  3. Claude/GPT answers with     │
│     retrieved context           │
└─────────────────────────────────┘
      ↓
PostgreSQL + pgvector
```

## 🔧 Tech Stack

- **Backend:** Python 3.11, FastAPI, SQLAlchemy async
- **Bot:** python-telegram-bot 21.x
- **AI:** OpenRouter (Claude Haiku / GPT-4o-mini)
- **Embeddings:** OpenAI text-embedding-3-small
- **Vector DB:** PostgreSQL + pgvector extension
- **Containerization:** Docker + docker-compose

## 🚀 Quick Start

```bash
# 1. Clone and configure
cp .env.example .env
# Fill in your API keys in .env

# 2. Run everything
docker-compose up -d

# 3. Upload a document
curl -X POST http://localhost:8000/upload \
  -F "file=@your_document.pdf" \
  -F "namespace=my_company"

# 4. Start chatting on Telegram
# Find your bot and send /start
```

## 📁 Project Structure

```
├── src/
│   ├── main.py          # FastAPI app + upload endpoint
│   ├── bot.py           # Telegram bot handlers
│   ├── rag.py           # RAG pipeline (embed, search, answer)
│   ├── db.py            # Database models + pgvector setup
│   └── config.py        # Settings from .env
├── documents/           # Sample PDFs for testing
├── scripts/
│   └── seed_demo.py     # Seeds DB with sample data
├── docker-compose.yml
├── Dockerfile
└── requirements.txt
```

## 💼 Use Cases (for clients)

- Customer support bot trained on your product docs
- Internal HR bot trained on company policies
- Real estate bot trained on property listings
- Legal bot trained on contracts / terms

## 📊 Performance

- Embedding: ~200ms per query
- Vector search: ~50ms (pgvector IVFFLAT index)
- Total response time: ~1.5-2.5s end-to-end
