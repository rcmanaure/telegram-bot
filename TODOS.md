# TODOS

## LLM Fallback Chain

- [ ] **FALL-1 (P2, fallback alert to operator)** — When `llm_failover` fires in `llm.py:call_chat()`, send a real-time Telegram message to `operator_chat_id` via the tenant's bot: `"⚠️ LLM fallback activated: {model} at {time}. Primary error: {error}"`. Deduped per hour per tenant to prevent alert spam. Piggybacks on the existing `daily_digest_job` pattern in `src/services/jobs.py`. Trigger: first time Ollama Cloud goes down in production and the operator misses it for >1h. Effort: CC ~20min / human ~1 day. Depends on: N-model chain implementation.

- [ ] **FALL-3 (P3, admin UI fallback format hint)** — After T1 ships, the `llm_fallback_model` field in the admin UI accepts comma-separated model names but shows no hint. Add `<small>Separate multiple models with commas: model1,model2</small>` under the input in `src/templates/admin/index.html`. Trigger: first operator reports confusion seeing the comma-separated value. Effort: CC ~5min / human ~10min.

- [ ] **FALL-2 (P3, add paid model to fallback chain when revenue allows)** — Add one paid model (e.g., `anthropic/claude-haiku-4-5` via OpenRouter ~$0.25/MTok) as a 4th fallback after `xiaomi/mimo-v2.5`. Currently both fallbacks (openrouter/free + mimo-v2.5) share OpenRouter as provider — a single OpenRouter outage takes out all fallbacks. A paid model on a separate provider (Anthropic-backed) breaks the concentration. The N-model chain already supports this as a zero-code `.env` addition. Trigger: first paying client complains about downtime from an OpenRouter outage, or monthly LLM budget allows ~$15-20/month. Effort: CC ~5min (env var) / human ~30min (verify + test). Context: accepted as deferred during /plan-ceo-review 2026-06-04.

## Localization — Spanish Dialect

- [x] **DIAL-0 (P1, neutral LATAM default)** — Added "use tú/usted/ustedes, never vosotros" instruction to `build_system_prompt()` and `_triage_response()` system prompts. Tests T2+T3 verify instruction presence. Shipped as part of localization groundwork.

- [ ] **DIAL-1 (P2, per-tenant dialect config)** — Add `dialect` nullable column to `Tenant` model (enum: `neutral_latam` default, `rioplatense`, `usted_formal`). Alembic migration. Admin UI select per tenant. `build_system_prompt()` takes `dialect=` param, branches the dialect instruction accordingly. Trigger: first client from Argentina/Uruguay requests voseo, or first Colombian client requests formal usted. Effort: CC ~1h / human ~2 days. Context: Approach B from design doc `~/.gstack/projects/rcmanaure-telegram-bot/root-main-design-20260604-142108.md`. Current neutral LATAM platform default ships as DIAL-0 (this PR). Alembic migration pattern follows existing nullable Tenant columns.

- [ ] **DIAL-2 (P3, align triage strings to neutral LATAM)** — Hardcoded voseo strings in `src/rag.py` bypass LLM and reach users directly: `"Entiendo que querés hablar con alguien"` (line 1435), `"No encontré información de precios"` (line 504), off_topic_reply in `prompts.py` line 22. Change these to neutral LATAM tú forms for voice consistency. Trigger: after DIAL-0 ships and a client notices the inconsistency. Effort: CC ~10min / human ~30min. Context: these strings are not LLM-generated — they bypass the dialect instruction entirely.

- [ ] **DIAL-3 (P3, eval test for LATAM dialect compliance)** — Add an integration eval test that sends real LLM calls to the deployed bot and asserts responses don't contain Spain Spanish markers (vosotros, ordenador, vale as filler). Unit tests (T2, T3) verify the instruction is present in prompts; this eval proves the LLM obeys it. Requires LLM eval infrastructure (e.g., `pytest -m integration` with a mock LLM fixture or live test tenant). Trigger: when an eval harness exists or a dialect regression is reported. Effort: CC ~30min / human ~2h. Context: T2 tests prompts.py, T3 tests rag.py — but neither proves the LLM actually produces LATAM Spanish output.

## Tool Use Agent — Feature Backlog (post-v1)

- [ ] **TOOL-E1 (P2, source citations)** — After tool dispatch, extract chunk sources (`source` field) and web URLs from tool results; include in answer or append as footnotes. Trigger: first client asks "where did you get this?" or feedback shows distrust. Effort: CC ~30min / human ~2h. Context: tool results already carry chunk dicts — parsing is 10 lines. See CEO plan `~/.gstack/projects/rcmanaure-telegram-bot/ceo-plans/2026-06-03-tool-use-agent.md`.

- [ ] **TOOL-E3 (P3, admin tool telemetry)** — Log which tools were called per conversation turn (new column or JSON field on `Conversation`). Admin `/admin?tab=tools` showing search_docs vs search_web call counts, hit rate, top queries triggering each. Trigger: after 4+ weeks of prod data. Effort: CC ~1h / human ~1 day. Context: build from real data to know what metrics matter.

- [ ] **TOOL-E5 (P2, tenant tool registry)** — Add `tool_config JSONB` column to `Tenant` table. Keys: `search_docs_enabled` (bool), `search_web_enabled` (bool), future custom tools. Admin UI toggles per tool per tenant. Alembic migration. Trigger: first client requests a custom integration (inventory API, price lookup). Effort: CC ~2h / human ~3 days. Context: current `web_search_enabled` boolean migrates to `tool_config.search_web_enabled`.

- [ ] **TOOL-UX1 (P3, pre-message deletion on tool_use degradation)** — Store Telegram message_id of "Procesando..." pre-message. If `is_tool_use_available()` returns False after the pre-message fires (TOCTOU race window during backoff transitions), delete the status message before sending the actual answer. WhatsApp has no edit/delete API so WA side is accept-as-is. Trigger: first user complaint about misleading status message. Effort: CC ~30min / human ~1h. Context: cosmetic edge case affecting <1% of requests during backoff transitions; pre-message condition already guards the main window with `is_tool_use_available()` check.

## Multi-tenant SaaS — Semana 1-2

### Fundaciones (orden CRÍTICO — seguir en este orden)

- [x] **T15** `requirements.txt` — agregar: `alembic`, `slowapi`, `sentry-sdk[fastapi]`
- [x] **T1** `src/db.py:54-55` — eliminar `print(DATABASE_URL)` con credenciales en stdout
- [x] **T2** `alembic init alembic` + baseline migration manual + `alembic upgrade head`
- [x] **T3** `docker-compose.yml` — `uvicorn --workers 1` sin `--reload`, eliminar servicio `bot`
- [x] **T4** `src/config.py` — agregar `app_domain`, `sentry_dsn`, `environment`; eliminar `telegram_bot_token`, `default_namespace`
- [x] **T5+T6** `src/db.py` (schema) + Alembic migration `766462df7dd1_multi_tenant_schema`:
  - Tabla `Tenant` (id, slug, `api_key_hash`, `webhook_secret`, bot_token, plan, billing_id, created_at, active)
  - HNSW index en `document_chunks.embedding` (m=16, ef_construction=64)
  - `Conversation`: `telegram_user_id` → `user_id` + `channel VARCHAR(20) DEFAULT 'telegram'` + `tenant_id FK`
  - `src/rag.py`: renombrar parámetros `telegram_user_id` → `user_id` + actualizar SQL raw
  - `src/bot.py`: actualizar SQL raw `telegram_user_id` → `user_id`
- [x] **T7** `src/main.py` — `require_tenant(X-API-Key)` con SHA-256 hash; aplicar a `/upload`, `/stats`, `DELETE /namespace` (sin path param — borrar solo el propio namespace)
- [x] **T8** `src/main.py` + `src/bot.py` — webhook endpoint `POST /webhook/{tenant_slug}` + lifespan con Application init + `set_webhook()` per-tenant + refactor handlers para `ctx.bot_data["tenant"]` + `cmd_sources` via DB directo

### Seguridad crítica (parte del T8)

- [x] Lifespan: `try/except` por tenant en `set_webhook()` — un token inválido no debe crashear el servidor (**T13**)
- [x] Webhook: validar `tenant.webhook_secret` (no global env var) — secret por tenant en DB (**D9**)
- [x] `DELETE /namespace` sin path param: `namespace = tenant.slug` del tenant autenticado (**D10**)

### Observabilidad (orden flexible, después de T5-T8)

- [x] **T9** `src/logging_config.py` (nuevo) + reemplazar `print()` en `bot.py`, `main.py`, `rag.py`
- [x] **T10** `slowapi` — `@limiter.limit("10/minute")` en `/upload`, `/stats`; `@limiter.limit("20/minute")` en `/webhook`
- [x] **T11** Sentry SDK — `sentry_sdk.init()` en `main.py`
- [x] **T14** `main.py` lifespan — `await http_client.aclose()` en shutdown

### Tests

- [x] **T12** `tests/test_rag_pipeline.py` — fixture `authed_api_client` + 7 tests nuevos + corregir 3 regressions:
  - Actualizar: `test_upload_rejects_non_pdf`, `test_upload_rejects_oversized_file`, `test_upload_rejects_corrupted_pdf`, `test_upload_accepts_uppercase_extension` → usar `authed_api_client`
  - Agregar: `test_upload_requires_auth()`, `test_stats_requires_auth()`, `test_delete_namespace_requires_auth()`
  - Agregar: `test_webhook_rejects_invalid_signature()`, `test_webhook_returns_404_unknown_tenant()`
  - Agregar @integration: `test_cross_tenant_isolation()`, `test_delete_only_deletes_own_namespace()`

### Variables de entorno a agregar a `.env`

- [x] `APP_DOMAIN=` (ej: `mybotplatform.fly.dev`, o `xxxx.ngrok.io` para dev)
- [x] `SENTRY_DSN=` (vacío en dev)
- [x] `ENVIRONMENT=dev`
- [ ] Eliminar de `.env`: `TELEGRAM_BOT_TOKEN=`, `DATABASE_URL_SYNC=`, `DEFAULT_NAMESPACE=` (removidos del modelo — no se usan)

## Smart Chatbot v2 — Completado ✅

- [x] **B-1** `/contactar` command + 4-category `_triage_response` (returns `tuple[str,str]`)
- [x] **B-2** `UnansweredQuery` DB model + admin "Consultas sin respuesta" tab
- [x] **CP-1** InlineKeyboard URL button + `contact_url` on Tenant (+ http/https validation)
- [x] **CP-2** "¿Hay algo más?" post-process in `handle_message` (after save_turn, NOT in system prompt)
- [x] **CP-3** `example_questions` on Tenant + richer `/start` numbered list
- [x] **CP-4** APScheduler daily digest (AsyncIOScheduler, 8am UTC, frequency-sorted top 5) + `operator_chat_id` on Tenant
- [x] **CP-5** Language-aware triage (`language_code` flows: handle_message → rag_query → _triage_response)
- [x] **OPT-1** Pre-RAG regex shortcut for explicit escalation phrases (skip embed + pgvector)
- [x] **OPT-2** Weekly UnansweredQuery cleanup job (DELETE WHERE created_at < 90 days ago)
- [x] Alembic migration: 3 new Tenant columns + UnansweredQuery model + 2 indexes (`a1b2c3d4e5f6`)
- [x] Fix 5 broken test mocks + 15 new tests — 62/62 passing

## STT — Voice Notes (Groq Whisper)

- [x] **STT-1** Groq Whisper integration (core) — `src/config.py` + `src/rag.py:transcribe_voice` + `src/bot.py:handle_voice + _process_question` + `src/main.py` register `filters.VOICE`; see CEO plan `~/.gstack/projects/rcmanaure-telegram-bot/ceo-plans/2026-05-26-groq-stt.md`
- [ ] **STT-2** Per-tenant Groq API key (`Tenant.groq_api_key` nullable, admin UI field, falls back to global key) — trigger: first paying client hits quota or requests separate billing; Effort M (CC: ~10min)

## Admin UI Redesign + Secure Config ✅

- [x] **A1** `requirements.txt` — add `cryptography>=42.0.0`
- [x] **A2** `src/crypto.py` — new: `get_fernet()`, `encrypt_value()`, `decrypt_value()` (fallback plaintext), `generate_key()`
- [x] **A3** `src/config_overlay.py` — `_overlay`, `get_setting()`, `reload_from_db()` (per-row try/except)
- [x] **A4** `src/db.py` — add `SystemConfig(key, encrypted_value)` model
- [x] **A5** Alembic migration — `system_config` table + conversations composite index + tenant_id backfill
- [x] **A6** `src/llm.py` — replace `settings.x` with `get_setting(key, fallback)`; `embedding_dim` int-cast; `reset_embedding_client()`
- [x] **A7** `src/lifespan.py` — call `reload_from_db()` after `init_db()`; close LLM + STT clients on shutdown
- [x] **A8** `src/routes/admin.py` — new endpoints: `/admin/settings` (POST), `/admin/settings/test-connection`, `/admin/tenant/{id}/toggle-active`, `/admin/health-data`
- [x] **A9** `src/templates/admin/index.html` — tabbed redesign (?tab= URL param), card layout, credential masking
- [x] **A10** `src/db.py` — `@hybrid_property` on `wa_access_token`/`wa_app_secret` with encrypt/decrypt via `crypto.py`
- [x] **A11** `src/routes/admin.py:admin_update_tenant()` — hybrid_property encrypts WA creds on write
- [x] **A12** `.env.example` — add `ENCRYPTION_KEY=`, `LLM_API_KEY`, `EMBEDDING_API_KEY`, `ADMIN_PASSWORD`, `SENTRY_DSN`, `ENVIRONMENT`
- [x] **A13** Tests — conftest patches `config_overlay.reload_from_db`; SlowAPI per-tenant key func; existing 237 tests pass
- [ ] **KEY-ROTATION (P3, Deferred)** Script to re-encrypt all SystemConfig + Tenant WA creds when ENCRYPTION_KEY changes; no implementation until first key rotation request

## Vision + Ollama Cloud (próxima branch después de feat/whatsapp-multi-channel)

- [x] **PREREQ-WS** Validar endpoint Ollama cloud web search antes de V7 — `curl -H "Authorization: Bearer $OLLAMA_API_KEY" https://ollama.com/api/web_search -d '{"query":"test","num_results":3}'`. Si falla, usar Tavily o Brave Search. **Bloquea V7.** **Completed:** v0.2.0.0 (2026-05-29)
- [x] **V1** `config.py` + `requirements.txt` — `LLM_VISION_MODEL: str = ""`, `WEB_SEARCH_URL: str = ""`, agregar `filetype` **Completed:** v0.3.0.0 (2026-06-01)
- [x] **V1.5** `llm.py:call_chat()` — param `model: str | None = None` **Completed:** v0.3.0.0 (2026-06-01)
- [x] **V2** `rag.py:generate_answer()` — `image_b64: str | None = None`, content array OpenAI vision format: `[{type:text}, {type:image_url, image_url:{url:data:image/..;base64,...}}]` **Completed:** v0.3.0.0 (2026-06-01)
- [x] **V2.5** `rag.py:rag_query()` — `image_b64` + `tenant: Tenant | None` params **Completed:** v0.3.0.0 (2026-06-01)
- [x] **V3** `bot.py` — `handle_photo()` + `filters.PHOTO`; guard > 5MB; capturar httpx error → "No pude descargar la imagen." **Completed:** v0.3.0.0 (2026-06-01)
- [x] **V4** `main.py:_process_wa_message()` — remover early return para imagen; capturar errores de descarga y modelo **Completed:** v0.3.0.0 (2026-06-01)
- [x] **V5** Upload endpoint (async) — jpg/png detect; vision describe pre-step; guard desc > 100 chars **Completed:** v0.3.0.0 (2026-06-01)
- [x] **V6** Alembic migration — `Tenant.web_search_enabled boolean default false` (probar en DB limpia) **Completed:** v0.3.0.0 (2026-06-01)
- [x] **V7** `rag.py:rag_query()` — web search fallback con rescue total; **requiere PREREQ-WS** **Completed:** v0.3.0.0 (2026-06-01)
- [x] **V8** Admin UI — toggle web_search_enabled; image accept en file inputs; `.env.example` Ollama block **Completed:** v0.3.0.0 (2026-06-01)
- [x] **V10** Tests — handle_photo, vision generate_answer, image upload, web search triage **Completed:** v0.4.0.0 (2026-06-02)

## Deferred (cuando haya > 10 clientes)

- [x] Dynamic tenant reload sin restart — `POST /admin/tenant/{id}/toggle-active`
- [x] `slowapi` key_func usando API key hash en vez de IP — `_api_key_func` in `limiter.py`
- [ ] Redis para estado compartido en multi-instancia (habilita `--workers > 1`)
- [ ] Cambiar SHA-256 a `pbkdf2_hmac` si se permiten keys elegidas por el usuario (hoy usamos `secrets.token_urlsafe(32)`, no aplica)
- [ ] APScheduler → Celery + Redis para multi-worker (evita digests duplicados con workers > 1)
- [ ] Feedback thumbs up/down por respuesta
- [ ] Integración CRM
- [ ] Sugerencias proactivas de doc gaps ("¿Querés que agreguemos esta respuesta?")

## Validación de mercado (más importante que el código)

- [ ] Llamar a 3 dueños de pymes y mostrarles el bot demo en su propio teléfono
- [ ] Observar sin explicar — anotar cada momento de confusión
- [ ] Preguntar cuánto pagarían (esto es lo que nadie pregunta)
