# TODOS

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

## Deferred (cuando haya > 10 clientes)

- [ ] Dynamic tenant reload sin restart (`POST /admin/tenants/{slug}/activate`)
- [ ] `slowapi` key_func usando `tenant.id` en vez de IP
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
