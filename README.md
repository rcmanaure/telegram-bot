# RAG Bot — Multi-tenant Telegram + FastAPI + pgvector

A production-ready AI chatbot platform. Upload documents per client, each gets their own Telegram bot that answers questions exclusively from their documents.

Built as a portfolio demo by **Ruben C.** — AI Engineer & Backend Developer

## Demo flow

1. Create a tenant via the Admin UI (`/admin`)
2. Upload documents through the Admin UI or REST API
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
      ├── 3. Similarity threshold filter (MIN_SIMILARITY = 0.20)
      │     ├── Context found → answer from documents
      │     └── No context → LLM triage (greeting? off-topic? needs human?)
      ├── 4. LLM fallback: primary model fails → retry with fallback model
      └── 5. Answer with per-tenant system prompt + expertise area
      │
      ▼
PostgreSQL + pgvector
```

**Local dev:**
```
Docker Compose
├── postgres     pgvector/pgvector:pg16
├── ngrok        ngrok/ngrok:latest   ← public HTTPS tunnel (dev profile)
└── api          FastAPI + uvicorn (1 worker, async)
```

**Production (VPS):**
```
Docker Compose + Traefik
├── postgres     pgvector/pgvector:pg16       ← restart: unless-stopped
└── api          FastAPI + uvicorn (1 worker)  ← restart: unless-stopped
                      │
                      ▼
                Traefik (reverse proxy, Let's Encrypt, HTTPS)
                      │
                      ▼
                telegram webhook → /webhook/{slug}
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python 3.11, FastAPI, SQLAlchemy async |
| Bot | python-telegram-bot 21.x, webhook mode only |
| AI | OpenRouter — primary + fallback model, `text-embedding-3-small` embeddings |
| Vector DB | PostgreSQL 16 + pgvector (HNSW index) |
| Migrations | Alembic (async, runs on container start) |
| STT | Groq Whisper (`whisper-large-v3-turbo`) for voice notes |
| Rate limiting | slowapi (HTTP) + in-memory per-user (Telegram) |
| Observability | Sentry SDK (FastAPI + SQLAlchemy integrations) |
| Infra | Docker Compose (dev) / Traefik + Let's Encrypt (prod) |

---

## Quick Start

### Local development

```bash
# 1. Configure environment
cp .env.example .env
# Edit .env — set OPENROUTER_API_KEY, NGROK_AUTHTOKEN, ADMIN_PASSWORD

# 2. Start all services (includes ngrok for Telegram webhooks)
docker compose --profile dev up -d

# 3. Open admin panel
open http://localhost:8000/admin
```

The `docker-compose.override.yml` is auto-loaded in local dev. It adds:
- ngrok tunnel for Telegram webhooks
- Host port `127.0.0.1:8000:8000` for local access
- Source volume mounts for hot-reload (`./src`, `./tests`, etc.)
- PostgreSQL port 5432 exposed for debugging

### Production (VPS)

```bash
# 1. On the server
git clone <repo> && cd telegram-bot
cp .env.example .env
# Edit .env — set OPENROUTER_API_KEY, APP_DOMAIN=your-domain.com,
#              TRAEFIK_HOST=your-domain.com, ADMIN_PASSWORD

# 2. Deploy (override file is NOT used — only docker-compose.yml)
docker compose -f docker-compose.yml up -d --build

# 3. Create your first tenant
docker compose exec api python scripts/create_tenant.py
```

Production `docker-compose.yml` has:
- No ngrok (Traefik handles HTTPS)
- No host port mapping (Traefik routes via Docker network)
- No dev volume mounts (container runs from built image)
- `restart: unless-stopped` on all services

---

## Creating a Tenant

### Via Admin UI (recommended)

Open `https://your-domain.com/admin` (user: `admin`, password: `ADMIN_PASSWORD`).

Fill in the form:
- **Slug** — unique identifier, e.g. `mi-empresa`
- **Telegram Bot Token** — from [@BotFather](https://t.me/BotFather)
- **Área de expertise** — shown to users when they ask off-topic questions
- **Plan** — free / basic / pro
- **URL de contacto** — WhatsApp link or contact page
- **Chat ID del operador** — for daily digest of unanswered questions
- **Preguntas de ejemplo** — up to 5 suggested questions for `/start`

Save the API Key shown (only displayed once).

### Via CLI

```bash
docker compose exec api python scripts/create_tenant.py
```

---

## Uploading Documents

### Via Admin UI

In the tenant table, click "Subir documento" next to any tenant. Accepts PDF, Markdown (`.md`), and plain text (`.txt`).

Download the **template** (`/admin/template`) — a generic fill-in guide for any business.

### Via API

```bash
# PDF
curl -X POST https://your-domain.com/upload \
  -H "X-API-Key: YOUR_API_KEY" \
  -F "file=@your_document.pdf"

# Markdown
curl -X POST https://your-domain.com/upload \
  -H "X-API-Key: YOUR_API_KEY" \
  -F "file=@knowledge_base.md"
```

Supported formats: **PDF**, **Markdown** (`.md`), **plain text** (`.txt`). Max 10 MB per file.

---

## LLM Fallback

The system uses two models:

1. **Primary** (`LLM_MODEL`, default: `openrouter/free`) — tried first
2. **Fallback** (`LLM_FALLBACK_MODEL`, default: `openrouter/owl-alpha`) — used when primary fails

Fallback triggers on: rate limit (429), timeout, network error, or malformed response. Set `LLM_FALLBACK_MODEL=` (empty) to disable.

---

## Workers and Scaling

**Use 1 uvicorn worker.** Do not increase workers.

Three in-memory states break with multiple workers:
- `telegram_apps` dict — each worker gets its own copy; webhook hits wrong worker = silent failure
- Rate limit dict — per-process, not per-user; users bypass limits across workers
- APScheduler — duplicate cron jobs per worker

Uvicorn async handles hundreds of concurrent requests in a single process. The bottleneck is the external LLM API, not Python. If you need to scale beyond 1 worker, migrate state to Redis/DB first.

---

## Project Structure

```
├── src/
│   ├── main.py            # FastAPI app — endpoints, lifespan, admin UI
│   ├── bot.py             # Telegram handlers (webhook mode, per-tenant context)
│   ├── rag.py             # RAG pipeline: embed → search → triage → answer, LLM fallback
│   ├── db.py              # SQLAlchemy models (Tenant, DocumentChunk, Conversation, UnansweredQuery)
│   ├── config.py          # Pydantic settings from .env
│   ├── security.py        # Input sanitization, canary token, output validation
│   └── logging_config.py  # Structured stdout logging
├── alembic/
│   ├── env.py
│   └── versions/
│       ├── af07e4f0ea14_baseline.py            # Initial schema
│       ├── 766462df7dd1_multi_tenant_schema.py # Tenants table + HNSW index
│       ├── 3f8a1c2e9d47_add_expertise_area.py  # Per-tenant bot persona
│       └── xxxx_add_smart_chatbot_v2.py         # contact_url, example_questions, unanswered_queries
├── documents/
│   └── plantilla.md        # Generic document template for new tenants
├── scripts/
│   ├── create_tenant.py    # CLI alternative to admin UI
│   └── seed_demo.py        # Seed demo data
├── tests/
│   ├── test_rag_pipeline.py
│   ├── test_edge_cases.py
│   └── test_security.py
├── docker-compose.yml        # Production config (clean, no dev services)
├── docker-compose.override.yml # Dev overrides (auto-loaded locally)
├── Dockerfile
└── requirements.txt
```

---

## API Reference

All endpoints (except `/health`, `/admin`, `/webhook/*`) require `X-API-Key` header.

| Method | Endpoint | Auth | Description |
|---|---|---|---|
| GET | `/health` | — | Service health + model info |
| POST | `/upload` | API Key | Upload PDF/MD/TXT (max 10 MB, 10 req/min) |
| GET | `/stats` | API Key | Indexed documents for this tenant |
| DELETE | `/namespace` | API Key | Delete all documents for this tenant |
| PATCH | `/tenant` | API Key | Update `expertise_area`, `contact_url` |
| POST | `/webhook/{slug}` | Telegram secret | Telegram update handler (20 req/min) |
| GET | `/admin` | HTTP Basic | Admin UI — manage tenants, upload/delete docs |
| POST | `/admin/tenants` | HTTP Basic | Create new tenant |
| POST | `/admin/tenant/{id}` | HTTP Basic | Update tenant config |
| POST | `/admin/upload/{id}` | HTTP Basic | Upload document for tenant |
| POST | `/admin/delete-docs/{id}` | HTTP Basic | Delete all docs + conversations for tenant |
| GET | `/admin/template` | HTTP Basic | Download document template (Markdown) |
| GET | `/admin/queries/{id}` | HTTP Basic | View unanswered questions for tenant |
| GET | `/docs` | — | Swagger UI |

---

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `OPENROUTER_API_KEY` | Yes | — | Covers LLM + embeddings |
| `DATABASE_URL` | Yes | — | `postgresql+asyncpg://...` |
| `LLM_MODEL` | No | `openrouter/free` | Primary LLM model |
| `LLM_FALLBACK_MODEL` | No | `openrouter/owl-alpha` | Fallback on primary failure; empty = disabled |
| `EMBEDDING_MODEL` | No | `openai/text-embedding-3-small` | Via OpenRouter |
| `EMBEDDING_DIM` | No | `1536` | Must match model output |
| `APP_DOMAIN` | No | `localhost:8000` | Dev: leave default. Prod: your domain |
| `TRAEFIK_HOST` | No | — | Prod: same as APP_DOMAIN (Traefik routing) |
| `ADMIN_PASSWORD` | No | `changeme` | Admin UI password |
| `GROQ_API_KEY` | No | — | Required for voice note transcription |
| `NGROK_AUTHTOKEN` | No | — | Required in local dev for webhooks |
| `SENTRY_DSN` | No | — | Sentry error tracking |
| `ENVIRONMENT` | No | `dev` | Passed to Sentry |
| `CHUNK_SIZE` | No | `500` | Characters per document chunk |
| `CHUNK_OVERLAP` | No | `50` | Overlap between chunks |
| `TOP_K_RESULTS` | No | `4` | Chunks retrieved per query |

---

## Multi-tenant Model

Each **Tenant** has:
- Isolated document namespace (`slug`)
- Hashed API key (SHA-256, plaintext never stored)
- Per-message webhook secret (validated with `hmac.compare_digest`)
- Own Telegram bot token and Application instance (in-process)
- Configurable `expertise_area` — drives the bot's off-topic redirect and system prompt
- `contact_url` — WhatsApp link or contact page for escalation
- `example_questions` — suggested questions shown in `/start`
- `operator_chat_id` — receives daily digest of unanswered questions

Cross-tenant isolation enforced at the auth boundary. Integration test verifies no namespace leakage.

---

## RAG Guardrails

Two layers prevent the bot from answering outside its knowledge base:

1. **Similarity threshold** (`MIN_SIMILARITY = 0.20`): chunks below this score are discarded before reaching the LLM. Off-topic questions (math, coding, etc.) score 0.05–0.15 and never reach the model.

2. **LLM triage**: when no relevant context is found, a lightweight LLM call classifies the message as `greeting`, `off_topic`, `needs_human`, or `ambiguous`. Escalation keywords (e.g. "quiero hablar con un humano") short-circuit directly to the human handoff flow.

3. **Canary token**: a unique hex string embedded in the system prompt per server start. If the LLM leaks it in output, `validate_output()` blocks the message.

4. **Input sanitization**: NFC normalization, regex injection detection, 2000-char truncation (full scan before truncation). Chunks are scanned at index time too — injected chunks are skipped.

---

## Bot Commands

| Command | Description |
|---|---|
| `/start` | Welcome message with example questions |
| `/help` | Same as `/start` |
| `/sources` | List indexed documents for this tenant |
| `/clear` | Reset conversation history |
| `/contactar` | Open contact link (if configured) |

Voice notes are transcribed via Groq Whisper and processed as text queries.

---

## Tests

```bash
# Unit tests (no database required)
.venv/Scripts/python.exe -m pytest tests/ -v -k "not integration"

# Integration tests (requires docker compose up)
docker compose exec api python -m pytest tests/ -v -m integration
```

Tests use `app.dependency_overrides` — no live Telegram connection required.

---

## Deployment

### Production (VPS with Traefik)

```bash
# .env on the server
APP_DOMAIN=srv1546906.hstgr.cloud
TRAEFIK_HOST=srv1546906.hstgr.cloud

# Deploy — override file is NOT loaded (no local dev services)
docker compose -f docker-compose.yml up -d --build

# Create tenant
docker compose exec api python scripts/create_tenant.py
```

Traefik handles HTTPS via Let's Encrypt. No host port needed — routing is via Docker network labels.

### Local development

```bash
# .env for local dev
APP_DOMAIN=localhost:8000
NGROK_AUTHTOKEN=your_token

# Start with ngrok tunnel
docker compose --profile dev up -d

# Hot-reload: src/ changes are mounted as volume
```

The `docker-compose.override.yml` is auto-loaded by `docker compose up` and adds ngrok, localhost port, and dev volumes. It does NOT exist on the VPS.

---

## Use Cases

- Customer support bot trained on product documentation
- Internal HR bot trained on company policies
- Real estate bot trained on property listings
- Legal bot trained on contracts and terms
- Lab or clinic bot trained on services and FAQs
- Any business that wants a Telegram bot answering from their own documents