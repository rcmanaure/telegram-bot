# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Multi-tenant SaaS RAG chatbot platform. Each tenant gets an isolated Telegram bot that answers exclusively from their uploaded documents.

## Stack
- **Runtime**: Python 3.11, FastAPI, uvicorn (single worker, no --reload in prod)
- **Bot**: python-telegram-bot 21.x, webhook mode only (not polling)
- **DB**: PostgreSQL 16 + pgvector (HNSW index), async via SQLAlchemy + asyncpg
- **Migrations**: Alembic (async, `alembic upgrade head` runs on container start)
- **LLM/Embeddings**: OpenRouter API (OpenAI-compatible SDK). Primary model `openrouter/free`, fallback `openrouter/owl-alpha`
- **STT**: Groq Whisper (`whisper-large-v3-turbo`) for voice notes
- **Infra**: Docker Compose (api + postgres + ngrok dev profile). Prod: Traefik + Let's Encrypt

## Commands

```bash
# Run all unit tests
.venv/Scripts/python.exe -m pytest tests/ -v -k "not integration"

# Run single test file
.venv/Scripts/python.exe -m pytest tests/test_rag_pipeline.py -v

# Run single test by name
.venv/Scripts/python.exe -m pytest tests/ -v -k "test_call_llm_fallback_on_429"

# Integration tests (requires docker compose up)
docker compose exec api python -m pytest tests/ -v -m integration

# Run app locally (after setting .env)
.venv/Scripts/python.exe -m uvicorn src.main:app --workers 1 --port 8000
```

## Architecture

### Request flow
```
Telegram webhook POST /webhook/{tenant_slug}
  → validate webhook secret (hmac.compare_digest)
  → sanitize_user_input (injection detection)
  → _process_question
    → rate limit check (20 msg/60s per tenant:user)
    → rag_query()
      1. Escalation shortcut: "quiero hablar con un humano" → skip RAG, return needs_human
      2. Embed query → pgvector cosine search (WHERE namespace = tenant.slug)
      3. Drop chunks below MIN_SIMILARITY (0.20)
      4. No context → _triage_response() (LLM classifies greeting/off_topic/needs_human/ambiguous)
      5. Has context → generate_answer() (system prompt + last 6 history turns + context + question)
      6. validate_output() (canary token leak detection)
      7. save_turn() + trim history to 50 rows per user
```

### Multi-tenancy
Tenant isolation at two layers:
- **Auth**: Each tenant has SHA-256 hashed API key, per-tenant webhook secret, own Telegram bot Application
- **Data**: All DocumentChunk/Conversation rows keyed by `namespace = tenant.slug`. Queries filter by namespace. Cross-tenant isolation verified by integration test.

### LLM fallback
`_call_llm()` in rag.py tries primary model, then `llm_fallback_model` on any failure (429, timeout, network, malformed response). Logs WARNING on fallback use. Set `LLM_FALLBACK_MODEL=""` to disable.

### Security
- Canary token: `secrets.token_hex(8)` per server start, embedded in system prompt, checked in output validation, redacted from conversation history
- Input sanitization: NFC normalization, regex injection detection, 2000-char truncation (full scan before truncation)
- Chunk scanning: same injection patterns checked at index time, injected chunks skipped
- API key: SHA-256 hash stored, plaintext never persisted
- Single-worker constraint: canary token and rate limit dict are in-memory (no Redis yet)

### Key modules
- `src/rag.py` — RAG pipeline entry point `rag_query()`, `_call_llm()` (fallback), `chunk_text` (paragraph-aware), `embed_texts`, `retrieve_context`, `generate_answer`, `_triage_response`, `transcribe_voice`
- `src/bot.py` — Telegram handlers, per-user rate limiting (`defaultdict(deque)`)
- `src/main.py` — FastAPI app, lifespan (tenant loading + webhook registration), all HTTP endpoints, admin UI, APScheduler jobs
- `src/db.py` — Tenant, DocumentChunk, Conversation, UnansweredQuery models; `init_db()` (pgvector extension)
- `src/security.py` — `sanitize_user_input`, `scan_chunk_for_injection`, `validate_output`, `CANARY_TOKEN`
- `src/config.py` — Pydantic Settings (all env vars)

### Database
4 Alembic migrations: baseline → multi-tenant (tenants table + HNSW index) → expertise_area → smart chatbot v2 (contact_url, example_questions, unanswered_queries). Run with `alembic upgrade head` before app start.

## Git workflow
Always create a feature branch before coding. Run tests before merging to main. No direct pushes to main.

## Skill routing

When the user's request matches an available skill, invoke it via the Skill tool. When in doubt, invoke the skill.

- Product ideas/brainstorming → /office-hours
- Strategy/scope → /plan-ceo-review
- Architecture → /plan-eng-review
- Design system/plan review → /design-consultation or /plan-design-review
- Full review pipeline → /autoplan
- Bugs/errors → /investigate
- QA/testing site behavior → /qa or /qa-only
- Code review/diff check → /review
- Ship/deploy/PR → /ship or /land-and-deploy
- Save progress → /context-save
- Resume context → /context-restore