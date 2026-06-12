# RAG Chatbot Platform — Multi-tenant, Multi-channel, Production-ready

**Python · FastAPI · PostgreSQL + pgvector · Telegram · WhatsApp · Docker**

A full-stack AI chatbot platform I built end-to-end. Each client (tenant) gets their own Telegram and/or WhatsApp bot that answers questions strictly from their uploaded documents. The system handles multi-tenant isolation, real-time messaging on two channels, a self-service web portal, and a production security model — all running on a single async worker.

> Built by **Ruben C.** — [rcmanaure@gmail.com](mailto:rcmanaure@gmail.com) · [GitHub](https://github.com/rcmanaure)

---

## Engineering Highlights

These are the non-trivial problems I solved that go beyond typical tutorials:

### 1. Multi-tenant RAG with per-namespace vector isolation
Each tenant's documents live in the same pgvector table, isolated by a `namespace` column. The HNSW index covers both embedding similarity and namespace, so vector search never leaks context between clients. Similarity thresholds are two-tiered (0.20 primary, 0.10 low-confidence fallback) — the LLM is told when it's working from weak matches.

### 2. Query reformulation before vector search
Follow-up questions like "¿y el precio?" are meaningless in isolation. Before embedding, I run a reformulation step that resolves pronouns and references against the last 3 conversation turns, producing a standalone query. Vector search runs on the reformulated query; the LLM still sees the original.

### 3. Tool-use path for parallel retrieval
When the LLM provider supports tool_use, the model decides which retrieval tools to call (document search, web search) and the results are dispatched in parallel. Falls through gracefully to sequential pipeline on provider incompatibility or empty results.

### 4. Stateful multi-channel webhook architecture
One process runs multiple Telegram bot instances (one per active tenant) registered as webhooks, plus a WhatsApp Cloud API handler. All state is in-process: bot registry, rate limits, APScheduler. Using 1 worker is a deliberate constraint — documented in detail — with a clear migration path to Redis if scale requires it.

### 5. Self-service tenant portal with Row-Level Security
Tenants log in at `/portal` with bcrypt passwords and JWT sessions. All DB queries run under PostgreSQL RLS policies: the tenant's ID is set as a session GUC (`app.current_tenant_id`), and the policies enforce it. Security review caught a GUC leak across requests — fixed by resetting the GUC in a `finally` block.

### 6. Runtime config overlay without restart
Operators can update LLM model, HNSW search quality, rate limits, and other settings through the admin UI. Changes land in a `SystemConfig` table and are read on every request via `get_setting()`, overriding `.env` values. Values are encrypted with Fernet AES-128-CBC.

### 7. Security model built from scratch
- API keys: SHA-256 hashed, never stored in plaintext
- Telegram webhooks: `hmac.compare_digest` on per-tenant secret token
- WhatsApp webhooks: HMAC-SHA256 of raw body, verified before parsing
- Portal: CSRF protection on all form submissions, path traversal guard on source names
- Input: NFC normalization, regex injection scan, 2000-char truncation — applied at message receipt and at document index time
- Canary token: unique hex string embedded in every system prompt; output is blocked if it leaks

### 8. Policy-aware retrieval (E7-E9)
Documents can contain `policy_statement` and `section_header` chunks typed at index time. These are always injected into the context when relevant, regardless of similarity score. Prevents the LLM from answering "how much does X cost?" without mentioning that X requires a prior appointment.

### 9. WhatsApp 24-hour service window enforcement
Meta's Cloud API blocks outbound messages outside a 24-hour user-initiated window. The platform tracks the last inbound message timestamp per user and sends a re-engagement template when the window expires, rather than silently failing.

### 10. 550+ tests with deep edge case coverage
Test suite covers RAG pipeline edge cases (similarity thresholds, reformulation, triage paths, policy injection), security (injection patterns, canary leakage, namespace isolation), portal auth flows (CSRF, JWT expiry, plan limits), WhatsApp adapter behavior, and tool-use paths. Integration tests run against a live Docker database.

---

## Architecture

```
Telegram / WhatsApp user
        │
        ▼
POST /webhook/{slug}[/whatsapp]
        │
        ├── Webhook signature validation (HMAC, per-tenant)
        ├── Rate limiting: 20 msg/60s per user per channel
        │
        ▼
RAG Pipeline (src/rag.py)
        │
        ├── 1. Greeting fast-path (regex — no LLM call)
        ├── 2. Query reformulation (pronoun resolution, last 3 turns)
        ├── 3. Embed query → pgvector HNSW cosine search
        │         namespace = tenant.slug (cross-tenant leak impossible)
        ├── 4. Similarity filter: ≥0.20 strong, 0.10–0.20 low-confidence
        ├── 5. Policy chunks always injected when typed chunks present
        ├── 6. No context → web search (optional) → LLM triage
        │         (greeting | off_topic | needs_human | ambiguous)
        ├── 7. Tool-use path: parallel doc + web via LLM tool_call
        └── 8. Generate answer → validate output → save turn → footer
        │
        ▼
PostgreSQL 16 + pgvector
(RLS active on portal queries, HNSW index on embeddings)
```

**Request flow for the self-service portal:**
```
Browser → JWT cookie → Portal route
                          │
                          ├── SET LOCAL app.current_tenant_id = {id}
                          ├── All queries filtered by RLS policy
                          └── GUC reset in finally block (prevent leak)
```

---

## Tech Stack

| Layer | Technology | Notes |
|---|---|---|
| Backend | Python 3.11, FastAPI, SQLAlchemy async | Single async worker, uvloop |
| Telegram | python-telegram-bot 21.x | Webhook mode, multi-instance in-process |
| WhatsApp | Meta Cloud API | HMAC verification, 24h window, dedup cache |
| AI — Chat | OpenAI-compatible endpoint | Primary + fallback model, tool_use support |
| AI — Embeddings | OpenAI-compatible endpoint | Configurable provider, dim validation at startup |
| AI — Vision | Optional vision model | Image queries + image-indexed uploads |
| AI — STT | Groq Whisper | Voice notes on Telegram |
| Vector DB | PostgreSQL 16 + pgvector | HNSW index, `hnsw.ef_search` tunable at runtime |
| Portal auth | bcrypt + JWT | RLS-enforced per-tenant DB isolation |
| Runtime config | Fernet-encrypted DB overlay | Override `.env` without restart |
| Migrations | Alembic async | Auto-runs on container start |
| Scheduler | APScheduler | Daily digest, weekly cleanup |
| Infra (dev) | Docker Compose + ngrok | Hot-reload, local webhook tunnel |
| Infra (prod) | Docker Compose + Traefik | Let's Encrypt, no host port mapping |

---

## Project Structure

```
src/
├── rag.py                # RAG pipeline — 44 functions, the core of the system
├── bot.py                # Telegram handlers
├── llm.py                # call_chat(), call_embeddings(), tool_use dispatch
├── db.py                 # 6 SQLAlchemy models with RLS support
├── config_overlay.py     # Runtime config: DB overrides .env without restart
├── crypto.py             # Fernet encryption for stored secrets
├── security.py           # Input sanitization, canary token, output validation
├── channels/
│   ├── protocol.py       # ChannelAdapter Protocol — Telegram and WA share one interface
│   ├── telegram.py       # TG-specific formatting and handlers
│   └── whatsapp.py       # WhatsAppAdapter, service window, message dedup
├── services/
│   ├── wa_processor.py   # WA RAG pipeline, reply formatting
│   ├── knowledge.py      # Document indexing: chunk, embed, upsert, delete
│   ├── upload.py         # MIME detection, vision description, file processing
│   ├── prompts.py        # System prompt builder, greeting pattern classifier
│   └── usage.py          # Plan-based metering and audit log
└── routes/
    ├── api.py            # REST API (upload, stats, tenant config)
    ├── webhook.py        # Telegram + WhatsApp webhook handlers
    ├── admin.py          # Admin UI (Jinja2, HTTP Basic)
    └── portal.py         # Self-service portal (Jinja2, JWT, RLS)

tests/                    # 550+ tests across 16 files
├── test_rag_pipeline.py  # Core RAG: similarity, triage, reformulation
├── test_edge_cases.py    # 3389 lines of edge cases
├── test_portal_edge_cases.py  # Portal: auth, CSRF, plan limits, RLS
├── test_security.py      # Injection, canary, namespace isolation
├── test_whatsapp_integration.py  # WA adapter, footer suppression, service window
└── ...
```

---

## Quick Start

### Local development

```bash
cp .env.example .env
# Set: LLM_BASE_URL, LLM_API_KEY, EMBEDDING_BASE_URL, NGROK_AUTHTOKEN,
#      ADMIN_PASSWORD, ENCRYPTION_KEY

docker compose --profile dev up -d
open http://localhost:8000/admin
```

`docker-compose.override.yml` adds ngrok tunnel, port mapping, and source volume mounts for hot-reload. Not present on the VPS.

### Production (VPS + Traefik)

```bash
cp .env.example .env
# Set: APP_DOMAIN=your-domain.com, TRAEFIK_HOST=your-domain.com, + above

docker compose -f docker-compose.yml up -d --build
docker compose exec api python scripts/create_tenant.py
```

### Generate ENCRYPTION_KEY

```bash
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
```

---

## Self-service Portal

Tenants log in at `/portal` with their own credentials. From the dashboard they can:
- Upload and delete documents (plan-gated: free=3 docs, basic=10, pro=50)
- Ask test questions against their knowledge base
- View document index status and monthly query usage

Security: JWT sessions (HS256, configurable expiry), CSRF tokens on all forms, path traversal guard on document names, Postgres RLS enforcing tenant isolation at the database layer.

---

## Tenant Management

### Via Admin UI (`/admin`, HTTP Basic)

Tenant fields:
- **Slug** — namespace identifier
- **Telegram Bot Token** — from [@BotFather](https://t.me/BotFather)
- **WhatsApp credentials** — phone number ID, access token, app secret, verify token
- **Área de expertise** — drives off-topic redirects and system prompt framing
- **Plan** — free / basic / pro
- **URL de contacto** — escalation button target
- **Chat ID del operador** — daily digest of unanswered questions
- **Preguntas de ejemplo** — suggested questions in `/start`

### Via CLI

```bash
docker compose exec api python scripts/create_tenant.py
```

---

## API Reference

| Method | Endpoint | Auth | Description |
|---|---|---|---|
| GET | `/health` | — | Service health |
| POST | `/upload` | API Key | Upload document (PDF/MD/TXT/CSV/XLSX/DOCX, max 10 MB) |
| GET | `/stats` | API Key | Indexed documents for this tenant |
| DELETE | `/namespace` | API Key | Delete all documents |
| PATCH | `/tenant` | API Key | Update expertise area, contact URL, etc. |
| POST | `/webhook/{slug}` | TG secret token | Telegram update handler |
| GET/POST | `/webhook/{slug}/whatsapp` | WA HMAC | WhatsApp verification + message handler |
| GET | `/admin` | HTTP Basic | Admin UI |
| GET | `/portal/dashboard` | JWT cookie | Self-service tenant portal |
| POST | `/portal/ask` | JWT cookie | Test query against knowledge base |
| GET | `/docs` | — | Swagger UI |

---

## Key Environment Variables

| Variable | Description |
|---|---|
| `DATABASE_URL` | `postgresql+asyncpg://...` |
| `LLM_BASE_URL` | OpenAI-compatible chat endpoint |
| `LLM_API_KEY` | LLM provider key |
| `LLM_MODEL` | Primary model (e.g. `gpt-4o-mini`) |
| `LLM_FALLBACK_MODEL` | Fallback on primary failure |
| `LLM_VISION_MODEL` | Vision model; empty = vision disabled |
| `EMBEDDING_BASE_URL` | OpenAI-compatible embeddings endpoint |
| `EMBEDDING_MODEL` | Embeddings model |
| `EMBEDDING_DIM` | Must match model output (default: 1536) |
| `ENCRYPTION_KEY` | Fernet key for SystemConfig encryption |
| `APP_DOMAIN` | Prod: your domain. Dev: `localhost:8000` |
| `ADMIN_PASSWORD` | Admin UI password |
| `GROQ_API_KEY` | Voice note transcription (optional) |
| `NGROK_AUTHTOKEN` | Local dev webhook tunnel (optional) |
| `WEB_SEARCH_URL` | OpenAI-compatible web search (optional) |

---

## Worker Constraint

**Run exactly 1 uvicorn worker.** Three in-memory state objects break with multiple workers: the `telegram_apps` bot registry, per-user rate limit counters, and APScheduler. All documented in `CLAUDE.md`. Migration path to Redis is clear if scale requires it — this is a deliberate choice, not an oversight.

---

## Tests

```bash
# Unit tests (no database required, ~30 seconds)
.venv/Scripts/python.exe -m pytest tests/ -v -k "not integration"

# Integration tests (requires docker compose up)
docker compose exec api python -m pytest tests/ -v -m integration
```

550+ tests covering: RAG pipeline correctness, similarity threshold edge cases, query reformulation, triage classification, portal auth flows, CSRF, RLS enforcement, WhatsApp adapter behavior, source footer suppression by intent, vision paths, web search fallback, tool-use orchestration, security injection patterns, and namespace isolation.
