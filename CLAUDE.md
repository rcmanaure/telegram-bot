# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## gstack

Use `/browse` skill from gstack for all web browsing. Never use `mcp__claude-in-chrome__*` tools.

Available gstack skills: `/office-hours`, `/plan-ceo-review`, `/plan-eng-review`, `/plan-design-review`, `/design-consultation`, `/design-shotgun`, `/design-html`, `/review`, `/ship`, `/land-and-deploy`, `/canary`, `/benchmark`, `/browse`, `/connect-chrome`, `/qa`, `/qa-only`, `/design-review`, `/setup-browser-cookies`, `/setup-deploy`, `/setup-gbrain`, `/retro`, `/investigate`, `/document-release`, `/document-generate`, `/codex`, `/cso`, `/autoplan`, `/plan-devex-review`, `/devex-review`, `/careful`, `/freeze`, `/guard`, `/unfreeze`, `/gstack-upgrade`, `/learn`.

## Skill routing

When the user's request matches an available skill, invoke it via the Skill tool. When in doubt, invoke the skill.

Key routing rules:
- Product ideas/brainstorming → invoke /office-hours
- Strategy/scope → invoke /plan-ceo-review
- Architecture → invoke /plan-eng-review
- Design system/plan review → invoke /design-consultation or /plan-design-review
- Full review pipeline → invoke /autoplan
- Bugs/errors → invoke /investigate
- QA/testing site behavior → invoke /qa or /qa-only
- Code review/diff check → invoke /review
- Visual polish → invoke /design-review
- Ship/deploy/PR → invoke /ship or /land-and-deploy
- Save progress → invoke /context-save
- Resume context → invoke /context-restore
- Author a backlog-ready spec/issue → invoke /spec

## Commands

```bash
# Local dev — starts postgres + ngrok + api with hot-reload (requires NGROK_AUTHTOKEN in .env)
docker compose --profile dev up -d

# Production deploy (no ngrok, no host ports, Traefik handles HTTPS)
docker compose -f docker-compose.yml up -d --build

# Unit tests (no live services needed)
.venv/Scripts/python.exe -m pytest tests/ -v -k "not integration"

# Single test
.venv/Scripts/python.exe -m pytest tests/test_rag_pipeline.py::test_chunk_text_basic -v

# Integration tests (requires docker compose up first)
docker compose exec api python -m pytest tests/ -v -m integration

# Create a tenant interactively
docker compose exec api python scripts/create_tenant.py

# Apply DB migrations manually
docker compose exec api alembic upgrade head

# Generate ENCRYPTION_KEY for DB-stored secrets
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
```

`src/` is mounted as a volume in dev; changes hot-reload without rebuilding. `docker-compose.override.yml` is auto-loaded locally (exposes ports, mounts volumes, adds ngrok); delete or leave absent on the VPS for production.

## Architecture

**Multi-tenant** — each `Tenant` row has its own Telegram bot token (and optionally WhatsApp credentials). At startup, `lifespan()` in `src/lifespan.py` loads all active tenants, builds a `python-telegram-bot` `Application` per tenant, registers webhooks, and stores them in `telegram_apps: dict[str, Application]` in `src/state.py` (keyed by bot token). **Do not use more than 1 uvicorn worker** — this in-memory dict, rate limit state, and APScheduler all break with multiple processes.

**Main entry point** (`src/main.py` — 43 lines): slim wiring module that creates the FastAPI app, mounts static files, includes route routers, and registers the rate-limit exception handler. All business logic lives in extracted modules.

**Module map** (extracted from former 1267-line GOD service):
- `state.py` — in-memory `telegram_apps` registry + `register_app`/`get_app`/`update_tenant_cache`/`remove_app`
- `dependencies.py` — FastAPI dependencies: `require_tenant` (X-API-Key SHA-256 lookup)
- `limiter.py` — `RateLimiter` class (unified TG/WA, `time.monotonic()`, `>= MAX`), SlowAPI limiter + handler
- `lifespan.py` — startup/shutdown: DB init, ngrok discovery, tenant bot init, FAQ sync, APScheduler jobs
- `services/upload.py` — `detect_mime`, `describe_image_for_upload`, `process_uploaded_file`, `MAX_UPLOAD_BYTES`
- `services/ngrok.py` — `get_ngrok_domain()` with retry loop
- `services/jobs.py` — `daily_digest_job()`, `cleanup_job()` using `tg_rate_limiter.sweep()` + `wa_rate_limiter.sweep()`
- `services/tenant_bot.py` — `init_tenant_bot()` registers TG handlers + sets webhook
- `services/wa_processor.py` — `create_wa_adapter()`, `handle_wa_message()` (full WA RAG pipeline)
- `services/stt.py` — `transcribe_voice()` via Groq Whisper (own httpx client)
- `services/prompts.py` — `build_system_prompt()`, `ESCALATION_PATTERN`
- `routes/api.py` — `/health`, `/upload`, `/stats`, `/tenant` (PATCH), `/namespace` (DELETE)
- `routes/webhook.py` — `/webhook/{tenant_slug}` (TG), `/webhook/{tenant_slug}/whatsapp` (GET verify + POST)
- `routes/admin.py` — `/admin` panel (Jinja2 templates + static CSS), tenant CRUD, document upload/queries
- `src/channels/` — `protocol.py` (ChannelAdapter Protocol), `telegram.py` (handlers), `whatsapp.py` (WhatsAppAdapter)

**RAG pipeline** (`src/rag.py`):
1. `rag_query()` is the entry point, called by both Telegram and WhatsApp handlers
2. Escalation regex (from `services/prompts.py`) short-circuits before vector search for human-handoff phrases
3. `_reformulate_query()` rewrites follow-up questions into standalone queries (resolves pronouns from last 3 turns) before vector search; original question used for answer generation
4. `retrieve_context()` embeds the query → sets `hnsw.ef_search` + `hnsw.iterative_scan` via `SET LOCAL`, then cosine search via pgvector `<=>` operator filtered to `namespace = tenant.slug`. **Critical**: never `db.commit()` between the `SET LOCAL` statements and the `SELECT` — that ends the transaction and loses the settings.
5. Chunks below `MIN_SIMILARITY = 0.20` are dropped; second-pass at `LOW_MIN_SIMILARITY = 0.10` gives approximate matches (LLM notified via `low_confidence=True`). If both fail and images are present, `_extract_search_terms_from_images()` runs vision extraction and retries.
6. No context found → web search fallback (if `tenant.web_search_enabled`) → `_triage_response()` classifies as `greeting | off_topic | needs_human | ambiguous`
7. `generate_answer()` calls the LLM with `system → last-6-history → (context + question)`
8. `validate_output()` logs a warning if the canary token leaks; then `save_turn()` persists the exchange

Conversation history is auto-trimmed to `HISTORY_ROW_CAP = 50` rows per user+namespace. FAQ chunks are indexed under `source = "__faq__"` and re-synced whenever `example_questions` changes.

**LLM layer** (`src/llm.py`):
- `call_chat()` tries `LLM_MODEL` then `LLM_FALLBACK_MODEL`. When `model=` is passed explicitly (vision), only that model is tried — no fallback, since text-only fallback models cannot receive image payloads.
- `call_embeddings()` uses the OpenAI SDK against any OpenAI-compatible endpoint; chat and embedding providers can point to different bases (`LLM_BASE_URL` vs `EMBEDDING_BASE_URL`)
- `validate_config()` checks embedding dimensions at startup; `test_connection()` can verify a provider config manually

**Runtime config overlay** (`src/config_overlay.py` + `src/crypto.py`):
- `SystemConfig` DB table stores encrypted key-value pairs that override `.env` at runtime without restart
- `get_setting(key, fallback)` resolves: DB override → fallback (.env value); `get_setting_int(key, fallback)` for int settings (e.g. `hnsw_ef_search`)
- Values encrypted with Fernet AES-128-CBC (`src/crypto.py`); requires `ENCRYPTION_KEY` env var. If unset, values stored/read as plaintext with a warning. Decryption failure (e.g., value stored before encryption was enabled) falls back to plaintext silently.

**Contextual retrieval** (`LLM_CONTEXT_MODEL`): config key exists; `index_chunks()` does not yet use it. When implemented, a cheap model will prepend 50-100 token context summaries to each chunk *before* embedding (not stored in DB `content` column — original text stays clean for display).

**Channel abstraction** (`src/channels/`):
- `protocol.py` defines `ChannelAdapter` (Protocol), `ChannelMessage`, `ChannelSendError`, and per-channel formatting specs (`CHANNEL_FORMATTING`)
- `telegram.py` — handlers only; webhook dispatch lives in `routes/webhook.py`. Bot commands: `/start`, `/help`, `/sources`, `/clear`, `/contactar`
- `whatsapp.py` — `WhatsAppAdapter` implements the protocol; Meta's 24-hour service-window rule is enforced via `WaServiceWindow` table; messages are processed as `asyncio.create_task` (returns 200 immediately)
- Voice notes (Telegram): transcribed via Groq Whisper (`GROQ_API_KEY`); disabled fallback if key absent. WhatsApp voice: not yet implemented (returns placeholder).
- Images: both channels accept photos; passed as base64 to vision model (`LLM_VISION_MODEL`). Upload endpoint also accepts `.jpg/.jpeg/.png` — vision model describes the image, then text is indexed.

**DB models** (`src/db.py`): `Tenant`, `DocumentChunk` (with HNSW pgvector index), `Conversation`, `UnansweredQuery`, `WaServiceWindow`, `SystemConfig`.

**Migrations** (`alembic/`): async Alembic, runs automatically on container start (`alembic upgrade head`). Baseline migration uses `IF NOT EXISTS` — this hides wrong column types on fresh databases; only a `DROP TABLE` + re-run reveals type bugs.

**Security** (`src/security.py`): NFC normalization, regex injection scan on both inbound messages and document chunks at index time, 2000-char truncation, canary token in every system prompt.

**Document upsert pattern**: before inserting chunks, delete old rows with same `(namespace, source)` in the same transaction — keeps uploads idempotent. Compound index `ix_document_chunks_namespace_source` supports the `DELETE WHERE namespace = :ns AND source = :src` query.

**Conversation history poisoning risk**: old DB history is injected as few-shot examples into the LLM context. After major system prompt changes, clear history with `/clear` or `DELETE FROM conversations WHERE namespace = :ns`.

**Background jobs** (APScheduler, started in `lifespan()`):
- Daily digest at 08:00 UTC — sends top unanswered queries to `operator_chat_id` per tenant via Telegram
- Weekly cleanup (Sunday 00:00 UTC) — purges `UnansweredQuery` rows older than 90 days; sweeps stale rate-limit entries via `tg_rate_limiter.sweep()` + `wa_rate_limiter.sweep()`

**Admin UI** at `/admin`: HTTP Basic auth (`admin` / `ADMIN_PASSWORD` env var). Jinja2 templates with autoescape in `src/templates/admin/`, CSS in `src/static/css/admin.css`. Manage tenants, upload documents, view unanswered queries. Creating a tenant here also initializes the Telegram bot webhook in-process.

## Key constraints

- 1 uvicorn worker always — no Redis, so state is in-process
- `EMBEDDING_BASE_URL` must point to an OpenAI-compatible embeddings endpoint; Ollama Cloud does not support embeddings
- API key auth: SHA-256 hash stored, plaintext never stored; lookup is `WHERE api_key_hash = sha256(header)`
- Webhook auth: Telegram uses `X-Telegram-Bot-Api-Secret-Token` (`hmac.compare_digest`); WhatsApp uses HMAC-SHA256 of the raw body
- `telegram_apps` stores `Application` instances keyed by `bot_token` (not slug) — tenant-bot association goes through `tg_app.bot_data["tenant"]`
- Rate limits: 20 messages/60 s per user, unified `RateLimiter` class in `limiter.py` with `>= MAX` (was off-by-one `> MAX`)

## Testing

`tests/conftest.py` provides a **session-scoped** `TestClient` that patches `lifespan.init_db`, `db.AsyncSessionLocal`, and `services.ngrok.get_ngrok_domain`. Also sets `app.dependency_overrides[get_db]` with a mock DB. Use fixtures `api_client` (unauthenticated) or `authed_api_client` (overrides `require_tenant`) from conftest. Mock `rag.call_chat`, not HTTP internals, when testing RAG paths. When patching module-level functions, patch the module that **uses** the function (e.g., `patch("services.wa_processor.rag_query")`) not the module that defines it — because of `from X import Y` local references.

Test files: `test_rag_pipeline.py` (core RAG + API), `test_security.py`, `test_edge_cases.py`, `test_vision.py`, `test_web_search.py`, `test_whatsapp_integration.py`, `test_whatsapp_adapter.py`, `test_admin_redesign.py`, `test_image_buffer.py`.

## Plans

`PLAN_main_refactor.md` — T1–T15 implementation plan for extracting the former 1267-line GOD `main.py` into the current modular structure. Architecture decisions locked in that document (D1–D20). Consult before proposing structural changes to routes, services, or channel modules.

## graphify

This project has a knowledge graph at graphify-out/ with god nodes, community structure, and cross-file relationships.

Rules:
- For codebase questions, first run `graphify query "<question>"` when graphify-out/graph.json exists. Use `graphify path "<A>" "<B>"` for relationships and `graphify explain "<concept>"` for focused concepts. These return a scoped subgraph, usually much smaller than GRAPH_REPORT.md or raw grep output.
- If graphify-out/wiki/index.md exists, use it for broad navigation instead of raw source browsing.
- Read graphify-out/GRAPH_REPORT.md only for broad architecture review or when query/path/explain do not surface enough context.
- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost).
