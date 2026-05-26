# RAG Bot — Multi-tenant Telegram + FastAPI + pgvector

A production-ready AI chatbot platform. Upload PDFs per client, each gets their own Telegram bot that answers questions exclusively from their documents.

Built as a portfolio demo by **Ruben C.** — AI Engineer & Backend Developer

## Demo flow

1. Create a tenant via the Admin UI (`/admin`)
2. Upload PDFs through the REST API
3. Users chat with the Telegram bot — answers come only from the uploaded documents
4. Bot detects off-topic questions and redirects to the configured expertise area

---

## Architecture

```
User (Telegram)
      │
      ▼
Telegram Webhook  POST /webhook/{tenant_slug}
      │
      ▼
FastAPI Backend
      │
      ├── Auth: SHA-256 API key hash lookup (per tenant)
      ├── Webhook signature validation (per-tenant secret)
      │
      ▼
RAG Pipeline
      │
      ├── 1. Embed query (OpenRouter)
      ├── 2. pgvector HNSW cosine search (namespace = tenant slug)
      ├── 3. Similarity threshold filter (MIN_SIMILARITY = 0.30)
      │     ├── Context found → answer from documents
      │     └── No context → LLM triage (greeting? off-topic? redirect)
      └── 4. Answer with per-tenant system prompt + expertise area
      │
      ▼
PostgreSQL + pgvector
```

```
Docker Compose
├── postgres     pgvector/pgvector:pg16
├── ngrok        ngrok/ngrok:latest   ← public HTTPS tunnel, auto-discovered
└── api          FastAPI + uvicorn    ← Alembic migrations on startup
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python 3.11, FastAPI, SQLAlchemy async |
| Bot | python-telegram-bot 21.x, webhook mode |
| AI | OpenRouter — configurable LLM + `text-embedding-3-small` |
| Vector DB | PostgreSQL 16 + pgvector (HNSW index) |
| Migrations | Alembic (async) |
| Tunnel | ngrok Docker service (auto URL discovery) |
| Rate limiting | slowapi |
| Observability | Sentry SDK (FastAPI + SQLAlchemy integrations) |
| Infra | Docker Compose |

---

## Quick Start

### 1. Configure environment

```bash
cp .env.example .env
```

Edit `.env`:

```env
OPENROUTER_API_KEY=sk-or-...
DATABASE_URL=postgresql+asyncpg://ragbot:ragbot@postgres:5432/ragbot
LLM_MODEL=openrouter/owl-alpha
EMBEDDING_MODEL=openai/text-embedding-3-small
EMBEDDING_DIM=1536
NGROK_AUTHTOKEN=your_ngrok_token      # from dashboard.ngrok.com
ADMIN_PASSWORD=your_secure_password   # for /admin panel
```

### 2. Start all services

```bash
docker compose up -d
```

Startup sequence:
- postgres → healthy
- ngrok → tunnel established
- api → Alembic migrations → webhook registration → ready

### 3. Create your first tenant

Open `http://localhost:8000/admin` (user: `admin`, password: `ADMIN_PASSWORD`).

Fill in the form:
- **Slug** — unique identifier, e.g. `mi-empresa`
- **Telegram Bot Token** — from [@BotFather](https://t.me/BotFather)
- **Área de expertise** — shown to users when they ask off-topic questions
- **Plan** — free / basic / pro

Save the API Key shown (only displayed once).

### 4. Upload documents

Supported formats: **PDF**, **Markdown** (`.md`), **plain text** (`.txt`).

```bash
# PDF
curl -X POST https://your-ngrok-url/upload \
  -H "X-API-Key: YOUR_API_KEY" \
  -F "file=@your_document.pdf"

# Markdown
curl -X POST https://your-ngrok-url/upload \
  -H "X-API-Key: YOUR_API_KEY" \
  -F "file=@documents/acme_fitness.md"
```

A ready-to-use example is included at `documents/acme_fitness.md` (Acme Fitness Center FAQ).

### 5. Chat on Telegram

Find your bot and start asking questions about the uploaded documents.

---

## Project Structure

```
├── src/
│   ├── main.py            # FastAPI app — endpoints, lifespan, admin UI
│   ├── bot.py             # Telegram handlers (webhook mode, per-tenant context)
│   ├── rag.py             # RAG pipeline: embed → search → triage → answer
│   ├── db.py              # SQLAlchemy models (Tenant, DocumentChunk, Conversation)
│   ├── config.py          # Pydantic settings from .env
│   └── logging_config.py  # Structured stdout logging
├── alembic/
│   ├── env.py
│   └── versions/
│       ├── af07e4f0ea14_baseline.py            # Initial schema (IF NOT EXISTS)
│       ├── 766462df7dd1_multi_tenant_schema.py # Tenants table + HNSW index
│       └── 3f8a1c2e9d47_add_expertise_area.py  # Per-tenant bot persona
├── scripts/
│   └── create_tenant.py   # CLI alternative to admin UI
├── tests/
│   └── test_rag_pipeline.py
├── docker-compose.yml
├── Dockerfile
└── requirements.txt
```

---

## API Reference

All endpoints (except `/health`, `/admin`, `/webhook/*`) require `X-API-Key` header.

| Method | Endpoint | Auth | Description |
|---|---|---|---|
| GET | `/health` | — | Service health check |
| POST | `/upload` | API Key | Upload a PDF, Markdown or TXT file (max 10 MB, 10 req/min) |
| GET | `/stats` | API Key | Indexed documents for this tenant |
| DELETE | `/namespace` | API Key | Delete all documents for this tenant |
| PATCH | `/tenant` | API Key | Update `expertise_area` |
| POST | `/webhook/{slug}` | Telegram secret | Telegram update handler (20 req/min) |
| GET | `/admin` | HTTP Basic | Admin UI — manage all tenants |
| POST | `/admin/tenants` | HTTP Basic | Create new tenant |
| POST | `/admin/tenant/{id}` | HTTP Basic | Update tenant expertise area |
| GET | `/docs` | — | Swagger UI |

---

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `OPENROUTER_API_KEY` | Yes | — | Covers LLM + embeddings |
| `DATABASE_URL` | Yes | — | `postgresql+asyncpg://...` |
| `LLM_MODEL` | No | `openrouter/owl-alpha` | Any OpenRouter model |
| `EMBEDDING_MODEL` | No | `openai/text-embedding-3-small` | Via OpenRouter |
| `EMBEDDING_DIM` | No | `1536` | Must match model output |
| `NGROK_AUTHTOKEN` | Yes* | — | Required for webhook mode |
| `ADMIN_PASSWORD` | No | `changeme` | Admin UI password |
| `APP_DOMAIN` | No | `localhost:8000` | Fallback if ngrok unavailable |
| `SENTRY_DSN` | No | — | Sentry error tracking |
| `ENVIRONMENT` | No | `dev` | Passed to Sentry |
| `CHUNK_SIZE` | No | `500` | Characters per document chunk |
| `CHUNK_OVERLAP` | No | `50` | Overlap between chunks |
| `TOP_K_RESULTS` | No | `4` | Chunks retrieved per query |

*ngrok token only needed in dev/local. In production, set `APP_DOMAIN` to your real HTTPS domain instead.

---

## Multi-tenant Model

Each **Tenant** has:
- Isolated document namespace (`slug`)
- Hashed API key (SHA-256, never stored in plaintext)
- Per-message webhook secret (validated with `hmac.compare_digest`)
- Own Telegram bot token and Application instance (in-process)
- Configurable `expertise_area` — drives the bot's off-topic redirect message and system prompt

Tenants cannot access each other's documents. Namespace isolation is enforced at the auth boundary, not at query time.

---

## RAG Guardrails

Two layers prevent the bot from answering outside its knowledge base:

1. **Similarity threshold** (`MIN_SIMILARITY = 0.30`): chunks below this score are discarded before reaching the LLM. Off-topic questions (math, coding, etc.) score 0.05–0.15 against business documents and never reach the model.

2. **LLM triage**: when no relevant context is found, a lightweight LLM call classifies the message. Greetings get a warm response; genuine off-topic questions get redirected to the configured expertise area.

---

## Bot Commands

| Command | Description |
|---|---|
| `/start` | Welcome message |
| `/help` | Same as `/start` |
| `/sources` | List indexed documents for this tenant |
| `/clear` | Reset conversation history |

---

## Tests

```bash
docker compose exec api python -m pytest tests/ -v
```

Tests use `app.dependency_overrides` — no live database or Telegram connection required.

---

## Use Cases

- Customer support bot trained on product documentation
- Internal HR bot trained on company policies
- Real estate bot trained on property listings
- Legal bot trained on contracts and terms
- Any business that wants a Telegram bot answering from their own documents
