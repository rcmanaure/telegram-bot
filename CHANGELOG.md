# Changelog

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