# Changelog

## [0.6.2.0] - 2026-06-12

### Fixed

- **Source footer no longer shown on greetings** — Bots were appending "📎 Fuentes: …" to greeting responses like "¡Buen día!". The footer is now only shown when a response is grounded in retrieved documents (`intent = None`). Triage responses (greeting, off_topic, needs_human, ambiguous) never show sources regardless of what the retrieval pipeline returned.
- **Greeting classifier recognizes "buen día" and inverted punctuation** — `_GREETING_PATTERN` now matches singular `buen día`/`buen di a` and handles `¡`/`¿` prefix characters, so `¡buen día!` is correctly classified as a greeting rather than falling through to vector search.
- **26 new edge-case tests** — `TestGreetingPattern` (14 tests covering variant forms, punctuation, case insensitivity), plus Telegram and WhatsApp footer-suppression tests asserting that each intent path (greeting, off_topic, needs_human, doc-answer) produces or suppresses the footer correctly.

## [0.6.1.0] - 2026-06-11

### Changed

- **Clean response format** — Removed inline `[Source: X, Page Y]` citations from LLM responses. The system prompt no longer requires source references inside the answer text. Sources are still tracked via the footer and portal API, but patient-facing responses are now clean and readable without internal metadata.
- **Human-readable source names in footer** — `_build_source_footer()` now normalizes filenames: strips extensions (`.md`, `.pdf`, etc.), replaces hyphens/underscores with spaces, title-cases words. Footer shows `Sp Diagnostico Histologico` instead of `sp-diagnostico-histologico.md p.0`. Portal API still returns raw filenames in structured JSON.

## [0.6.0.0] - 2026-06-11

### Added

- **Self-service tenant portal** — tenants can now log in, manage documents, preview answers, view usage, and resolve unanswered questions through a browser dashboard at `/portal/dashboard`. Cookie-based auth with JWT, rate-limited login, and RLS-scoped DB sessions.
- **Row-Level Security (RLS)** — Postgres RLS policies on all tenant-scoped tables (`document_chunks`, `conversations`, `unanswered_queries`, `feedback`, `wa_service_windows`, `tenant_audit_log`, `tenant_usage`). Tenant sessions connect as `ragbot_tenant` with GUC `app.current_tenant` set per-request; admin engine bypasses RLS as table owner. `tenants` table also has RLS (`tenant_self` policy) so tenant-scoped sessions can only see their own row.
- **JWT portal authentication** — `src/auth.py` with bcrypt password hashing, `create_access_token`/`decode_access_token`, `require_portal_auth` dependency that reads JWT from `Authorization` header or `portal_token` cookie.
- **CSRF protection** — HMAC double-submit CSRF tokens for all state-changing portal forms. Bearer header auth is exempt.
- **Plan-based quotas (E2)** — `PLAN_LIMITS` config (free/basic/pro) enforcing document count, chunk count, and monthly query limits on portal uploads and queries. Atomic `check_and_increment_usage` prevents TOCTOU races.
- **Citation enforcement (E1)** — system prompt now requires `[Source: X, Page Y]` format for all document-sourced answers across all channels. Faithfulness self-check gates on low-confidence results, appending a caveat when claims aren't grounded in retrieved context.
- **Usage metering in TG/WA paths** — Telegram bot and WhatsApp webhook handlers now call `increment_usage` after every successful RAG query, not just the portal.
- **Audit trail (E6)** — `TenantAuditLog` records every knowledge mutation (upload, delete, query, login) with actor, action, and detail payload.
- **Answer preview (E3)** — `/portal/query` endpoint lets tenants test answers in the dashboard with citation sources, scoped to their own namespace.
- **Unanswered question resolution (E5)** — tenants can mark unanswered questions as resolved from the portal dashboard.
- **RAG generalization (E1-E6)** — chunk metadata (`metadata_` JSONB column), policy chunk retrieval, section sibling retrieval, catalog overview with LLM-generated formatting, and doc structure summaries.
- **E7 policy chunks / E8 prompt rule / E9 section siblings** — `classify_chunk_type()` tags chunks as `price_row`, `faq_answer`, `policy_statement`, `section_header`, or `general_info`; section emoji generation; sibling retrieval for context window padding.
- **Tenant-scoped DB engine** — `_build_tenant_engine()` creates a SQLAlchemy async engine connecting as `ragbot_tenant` using `make_url().set()` (safe against special characters in passwords). Falls back to admin engine in dev mode.
- **Background reindex** — `process_upload_async` offloads CPU-bound PDF parsing to a `ThreadPoolExecutor`, keeping the async event loop responsive.
- **Index status tracking** — `set_index_status`/`get_index_status` with 5-minute TTL shows `◐ procesando` and `● activo` badges in the portal dashboard.
- **57 new portal edge case tests** covering auth branches, security hardening, knowledge service, rate limiting, and RLS isolation.

### Changed

- `tenant_session()` now uses `SET` (not `SET LOCAL`) with explicit `RESET` on exit, preserving RLS GUC across commits within a request.
- Portal write routes use `auto_commit=False` with a single final `commit()`, keeping the RLS scope alive for multi-write operations.
- CSRF check in `/portal/query` now runs before usage metering, preventing credit consumption on invalid requests.
- Portal login form no longer has a dead 429 exception branch.
- Dead `_validate_csrf()` stub removed — all CSRF validation goes through `_check_csrf()`.
- `_index_status` bug fixed — was keyed by `(namespace, source)` tuples but looked up with bare string, always returning `None`. Now uses `get_index_status()` correctly.
- `DocumentChunk.metadata_` column type changed from `JSON` to `JSONB` (matching migration). `TenantUsage.value` gained `server_default="0"`.
- Portal token cookie now has `secure=True`, `SameSite=Strict`, `HttpOnly`.
- RLS policies on `tenant_audit_log` and `tenant_usage` split from single `FOR ALL` into separate `SELECT` and `INSERT` policies, preventing UPDATE/DELETE on audit and usage tables.
- `_build_tenant_engine()` uses `make_url().set()` instead of fragile `url.index("@")` string parsing.
- Migration password escaping (`replace("'", "''")`) for SQL injection defense-in-depth.
- `tenants` RLS policy migration fixed: removed duplicate GRANT SELECT (already in `d6e7f8g9h0i2`), removed incorrect REVOKE from downgrade that would break RLS subqueries.

### Fixed

- **C1** — Portal auth returns 303 redirect (not 401 JSON) for unauthenticated browser sessions.
- **C2** — JWT empty secret rejected: `decode_access_token` raises `ValueError` if `signing_secret` is empty; lifespan logs error if `JWT_SECRET` unset.
- **C3** — GUC leak prevention: `tenant_session` resets `app.current_tenant` in `finally` block with rollback before reset.
- **H1** — TOCTOU race in usage metering fixed with atomic `check_and_increment_usage`.
- **H2** — RLS GUC persistence: `SET` (not `SET LOCAL`) so GUC survives commits within a request.
- **H3** — Path traversal guard on `/portal/delete/{source}` rejects `/`, `\`, and `..`.
- **M1** — Source validation in portal upload rejects empty filenames and normalizes paths.
- **M3** — `_row_to_chunk_dict` uses `getattr(row, "metadata_", None)` (correct column name).
- **M7** — Missing trailing newlines added to migration files.

### Fixed

- RLS GUC reset on commit — `SET LOCAL` was used in `tenant_session()`, which resets on `COMMIT`, breaking tenant isolation for multi-write portal routes. Fixed by switching to `SET` with explicit `RESET` on exit.
- Broken `_index_status` lookup — dict keyed by `(namespace, source)` tuples but checked with bare string `s["source"] in _index_status`, always returning False. Fixed by using `get_index_status(tenant.slug, s["source"])`.
- SQL injection in migration password — f-string interpolation of `TENANT_DB_PASSWORD` into `CREATE ROLE ... PASSWORD '...'`. Fixed by escaping single quotes.
- Broad exception catch — `except (ValueError, Exception)` swallowed all errors in resolve_question. Fixed to `except ValueError`.
- JSON/JSONB model mismatch — model declared `JSON` but migration converts to `JSONB`. Fixed model to `JSONB`.
- Filename normalization bypass — raw `file.filename` compared against normalized DB names. Fixed by calling `normalize_source_name()` before comparison.

## [0.4.2.0] - 2026-06-04

### Added

- N-model LLM fallback chain — `LLM_FALLBACK_MODEL` now accepts comma-separated model names (e.g. `openrouter/free,xiaomi/mimo-v2.5`). Each model is tried in order on failure; primary uses a 60s timeout, fallbacks use 15s each to bound worst-case latency.
- `reasoning_content` support — when a reasoning model returns `content=null` but populates `reasoning_content`, the reasoning text is used instead of falling through to the next model.
- Actual resolved model logged — `llm_call` log includes the `model` field from the provider response (shows the real model behind `openrouter/free` meta-routing).

### Fixed

- Empty-string content (`content=""`) now falls through to the next model in the chain instead of being returned as a valid response.
- `/health` endpoint reports the runtime-overridden model names (via `get_setting()`) instead of the compile-time `.env` values.

## [0.4.1.0] - 2026-06-02

### Added

- Vision-augmented retrieval — when a user sends an image with no matching text context, the bot extracts key search terms from the image and retries the vector search, handling medical orders, lab results, and other document photos that would otherwise return "not found."
- Multi-image buffer with debounce — when a user sends multiple photos (Telegram albums or rapid WA messages), images are buffered for 2.5s and processed together in a single LLM call (max 5 images per flush).
- Web search fallback — when RAG finds no matching documents, the bot can optionally search the web via Ollama Cloud before falling back to triage. Per-tenant toggle (`web_search_enabled`) in admin UI.
- Low-confidence retrieval fallback — near-match queries that score between 10% and 20% similarity now get a prompt noting the partial match instead of a flat "not found."
- Partially legible image handling — when the vision model can read part of an image, the bot extracts readable text and notes illegible sections rather than rejecting the entire image.
- Illegible image detection — images the vision model cannot read produce a clear, helpful message suggesting better lighting or focus.
- Query reformulation — follow-up questions with pronouns (e.g., "¿cuánto cuesta eso?") are rewritten using conversation history for more accurate retrieval.
- Conversation history summarization — old history entries beyond 50 rows are summarized by the LLM to keep context within token limits while preserving meaning.
- Greeting shortcut — common greetings ("hola", "hey", "buenas") skip vector search and triage entirely, responding instantly.
- Expanded escalation patterns — "contactar" and "representante" now trigger human handoff.
- Spanish prompt injection patterns — regex guards block LLM-manipulation attempts in Spanish ("ignora instrucciones", "olvida tu rol", etc.).
- Feedback endpoint — `GET /feedback` returns recent feedback entries (max 500 per tenant).
- Markdown table chunking — uploaded documents with tables split each row into its own chunk with the header prepended, preventing rows from being lost in large chunks.
- Source name normalization — browser-added dupe suffixes like `(1)`, `_copy`, `_2` are stripped before upsert, preventing duplicate document entries.
- Orphaned chunk cleanup on startup — browser-dupe chunks left from interrupted uploads are removed automatically.
- WA image buffering — WhatsApp images are debounced and flushed with text captions, mirroring the Telegram multi-image experience.
- WA escalation buttons — off-topic and needs-human responses include a "Contactar" button linking to the tenant's contact URL.
- Channel formatting — universal `**bold**` and `__italic__` markdown conversion for both Telegram and WhatsApp channels.
- Web search integration tests — 14 new tests covering enabled/disabled, silent fallback, not invoked when context found, and 404/405 retry with generic format.

### Fixed

- Image buffer self-cancel bug — `_flush` no longer cancels its own asyncio task, preventing `CancelledError` on buffered image processing.
- Vision guard — no 404 error when `LLM_VISION_MODEL` is not configured; instead returns a clear "not available" message.
- LLM repeating greetings — prompt change prevents the model from starting every response with "¡Hola! Con gusto te ayudo."
- Off-topic responses — warm redirection instead of blunt "fuera de mi área" rejection.
- Triage false negatives — partial-match rules now override `off_topic` classification when the query is within the tenant's expertise area.
- Low-confidence near-matches — prompt guides LLM to offer relevant information from partially matching documents instead of saying "not found."
- Upload response returns normalized `source_name` instead of raw browser filename, matching what `/stats` and `DELETE /namespace` use.
- Sanitize guard on vision-extracted query terms — `ValueError` from injection detection is caught gracefully instead of crashing the pipeline.
- Duplicated illegible-image message logic extracted to `_illegible_fallback_msg()` helper — single source of truth for singular/plural messages.
- Duplicated WA reply-sending logic extracted to `_send_wa_reply()` helper — sources footer and escalation button construction no longer duplicated across `_wa_process_flushed` and `handle_wa_message`.
- Webhook returns 503 (not 200) when bot Application not registered — Telegram retries on 503 instead of silently dropping the message.

## [0.4.0.0] - 2026-06-02

### Added

- Vision-augmented retrieval — when a user sends an image with no matching text context, the bot extracts key search terms from the image using the vision model and retries the vector search with those terms. Handles medical orders, lab results, and other document photos that would otherwise return "not found."
- `VISION_EXTRACT_MAX_TOKENS` constant (80 tokens) for search-term extraction prompt.
- Sanitization guard on vision-extracted query terms — LLM-generated search terms now pass through `sanitize_user_input()` before vector search, consistent with all other user input paths.
- Case-insensitive and whitespace-normalized comparison between original query and vision-extracted query to avoid redundant searches.
- 6 new test cases covering vision-augmented retrieval edge cases (no vision model, empty result, whitespace result, multiple images, same-query skip, low-confidence fallback).

### Fixed

- Missing `save_turn` mock in vision guard test caused RuntimeWarning about unawaited coroutine.
- Magic number `80` in `_extract_search_terms_from_images` extracted to named constant `VISION_EXTRACT_MAX_TOKENS`.

## [0.3.0.0] - 2026-06-01

### Added

- Runtime config overlay (`src/config_overlay.py`) — `SystemConfig` DB table stores encrypted key-value pairs that override `.env` at runtime without restart. `get_setting(key, fallback)` resolves DB override → fallback.
- Fernet AES encryption for DB-stored secrets (`src/crypto.py`) — WA credentials and config overlay values encrypted at rest. Requires `ENCRYPTION_KEY` env var; falls back to plaintext with warning if unset.
- Tab-based admin UI — CSS-only tabs (Tenants, Configuración, Salud) via server-side `?tab=` routing. Tenant cards grid layout replaces table.
- Settings management endpoint — `POST /admin/settings` saves config overlay keys (LLM, embeddings, Groq, web search) with encryption and live reload.
- Test Connection endpoint — `POST /admin/settings/test-connection` verifies LLM/embedding provider connectivity from admin UI.
- Health data endpoint — `GET /admin/health-data` returns JSON with LLM, DB, tenant, and chunk stats for admin health tab.
- Tenant toggle-active — `POST /admin/tenant/{id}/toggle-active` activates/deactivates tenants without restart.
- SlowAPI per-tenant rate limiting — custom `_api_key_func` uses X-API-Key SHA-256 hash for authed routes, IP for unauthed.
- `httpx.AsyncHTTPTransport(retries=2)` for automatic transport-level retries in LLM chat client.
- `reset_embedding_client()` for live config reload without restart.
- `close_llm_clients()` for proper shutdown of chat + embedding HTTP clients.
- STT HTTP client shutdown in lifespan.
- Default admin password warning on startup.
- Conversation composite index `ix_conversations_ns_user_created` for faster history queries.
- Alembic migration: `system_config` table, conversations composite index, `tenant_id` backfill for Conversation and UnansweredQuery.
- Mobile responsive admin CSS with `@media` breakpoints.
- Viewport meta tag in admin base template.

### Fixed

- WA typing indicator sent `"type": "reaction"` instead of `"type": "typing"` — typing indicator never displayed.
- WA rate-limit check moved before `sanitize_user_input` — injection probes could bypass rate limiting.
- Silent `_triage_response` exception now logs warning with error details.
- `save_turn` now accepts and persists `tenant_id` for proper tenant association.
- `json.JSONDecodeError` added to `call_chat` except clause — malformed LLM responses no longer crash.
- `delete_namespace` and `admin_delete_docs` now also delete `unanswered_queries` rows (orphaned data).
- Admin routes now have SlowAPI rate-limit decorators (20/min GET, 5/min POST).
- Toggle-active template bug: was `<a href>` (GET) on POST route → changed to `<form method="post">`.
- Config overlay value `settings.embedding_dim` in `validate_config` success log → now uses `emb_dim` from overlay.

## [0.2.0.0] - 2026-05-29

### Added

- WhatsApp Cloud API channel adapter (`src/channels/whatsapp.py`) — parse_incoming for text/voice/image/button/interactive messages, HMAC-SHA256 webhook verification, hub.challenge GET verification, send_reply with button support, 24h service window tracker, message dedup, and template message support
- Channel-agnostic protocol (`src/channels/protocol.py`) — ChannelAdapter Protocol, ChannelMessage/ChannelButton/ChannelSendError dataclasses, per-channel ChannelFormatting specs (Telegram + WhatsApp), format_text_for_channel post-processor, normalize_phone utility
- Telegram adapter scaffold (`src/channels/telegram.py`) — Phase 1 stub implementing ChannelAdapter
- WhatsApp webhook endpoints in main.py — GET /webhook/{slug}/whatsapp for verification, POST /webhook/{slug}/whatsapp for message processing with asyncio.create_task background processing
- Per-channel formatting in RAG pipeline — _build_system_prompt accepts channel param, rag_query/generate_answer/save_turn propagate channel to store in Conversation.channel
- Admin UI for WhatsApp credentials — collapsible "WhatsApp Config" section with phone_number_id, access_token, app_secret, verify_token, reengagement_template; auto-enables "whatsapp" channel when credentials are provided
- Alembic migration for WA schema — 7 new columns on tenants (wa_phone_number_id, wa_access_token, wa_app_secret, wa_business_id, wa_verify_token, wa_reengagement_template, channels), new wa_service_windows table with unique constraint on (tenant_id, user_id)
- WA config env vars in Settings (wa_phone_number_id, wa_access_token, wa_app_secret, wa_business_id, wa_verify_token)
- 45 unit tests covering normalize_phone, parse_incoming, verify_webhook (body param + fallback), handle_verification (timing-safe), format_text_for_channel, _build_system_prompt channel param, message dedup, ChannelFormatting specs, WhatsAppAdapter.format_text, ChannelSendError, ChannelMessage

### Fixed

- P0: _process_wa_message now creates its own AsyncSession via AsyncSessionLocal() — the request-scoped session closes before background tasks finish
- P1: verify_webhook now accepts body bytes parameter — handler reads request.body() before calling verify_webhook so HMAC signature is computed correctly
- P1: Fixed broken `from bot import rate_limits` — replaced with local _wa_rate_limits dict since WA runs in async context separate from TG bot
- P1: Fixed missing `from rag import _log_unanswered` import — was called but never imported
- P2: handle_verification uses hmac.compare_digest for timing-safe verify_token comparison