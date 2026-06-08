# Graph Report - .  (2026-06-08)

## Corpus Check
- 73 files · ~70,314 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 1237 nodes · 2331 edges · 99 communities (72 shown, 27 thin omitted)
- Extraction: 72% EXTRACTED · 28% INFERRED · 0% AMBIGUOUS · INFERRED: 662 edges (avg confidence: 0.7)
- Token cost: 8,200 input · 1,850 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Telegram Bot Handlers|Telegram Bot Handlers]]
- [[_COMMUNITY_LLM Client Layer|LLM Client Layer]]
- [[_COMMUNITY_RAG Answer Generation|RAG Answer Generation]]
- [[_COMMUNITY_DB Models & Migrations|DB Models & Migrations]]
- [[_COMMUNITY_RAG Query Pipeline|RAG Query Pipeline]]
- [[_COMMUNITY_Admin UI & Config|Admin UI & Config]]
- [[_COMMUNITY_LLM Tool Use|LLM Tool Use]]
- [[_COMMUNITY_History & Edge Case Tests|History & Edge Case Tests]]
- [[_COMMUNITY_Tenant & Base Models|Tenant & Base Models]]
- [[_COMMUNITY_Web Search Tests|Web Search Tests]]
- [[_COMMUNITY_Security & Input Guard|Security & Input Guard]]
- [[_COMMUNITY_Channel Formatting|Channel Formatting]]
- [[_COMMUNITY_WhatsApp Adapter Tests|WhatsApp Adapter Tests]]
- [[_COMMUNITY_Prompts & RAG Module|Prompts & RAG Module]]
- [[_COMMUNITY_RAG Pipeline Tests|RAG Pipeline Tests]]
- [[_COMMUNITY_WhatsApp Integration Tests|WhatsApp Integration Tests]]
- [[_COMMUNITY_LLM Call & Mock Tests|LLM Call & Mock Tests]]
- [[_COMMUNITY_Channel Message Types|Channel Message Types]]
- [[_COMMUNITY_Tenant Knowledge Docs|Tenant Knowledge Docs]]
- [[_COMMUNITY_WhatsApp Processor|WhatsApp Processor]]
- [[_COMMUNITY_Crypto & Encryption|Crypto & Encryption]]
- [[_COMMUNITY_Test Infrastructure|Test Infrastructure]]
- [[_COMMUNITY_Channel Adapter Protocol|Channel Adapter Protocol]]
- [[_COMMUNITY_WhatsApp Design Rationale|WhatsApp Design Rationale]]
- [[_COMMUNITY_Document Chunking|Document Chunking]]
- [[_COMMUNITY_Document Upload|Document Upload]]
- [[_COMMUNITY_Rate Limiting|Rate Limiting]]
- [[_COMMUNITY_Image Buffer Tests|Image Buffer Tests]]
- [[_COMMUNITY_Admin Edge Case Tests|Admin Edge Case Tests]]
- [[_COMMUNITY_Channel Buttons|Channel Buttons]]
- [[_COMMUNITY_System Prompt Builder|System Prompt Builder]]
- [[_COMMUNITY_Multi-Tenant & Dev Tunnel|Multi-Tenant & Dev Tunnel]]
- [[_COMMUNITY_REST API Routes|REST API Routes]]
- [[_COMMUNITY_Image Buffer|Image Buffer]]
- [[_COMMUNITY_Channel Protocol Types|Channel Protocol Types]]
- [[_COMMUNITY_Message Send & Errors|Message Send & Errors]]
- [[_COMMUNITY_WA Service Window|WA Service Window]]
- [[_COMMUNITY_Tenant State Registry|Tenant State Registry]]
- [[_COMMUNITY_Image Debounce Changelog|Image Debounce Changelog]]
- [[_COMMUNITY_Image Buffer Entries|Image Buffer Entries]]
- [[_COMMUNITY_Vision Search Terms|Vision Search Terms]]
- [[_COMMUNITY_Response Validation|Response Validation]]
- [[_COMMUNITY_Embeddings Client|Embeddings Client]]
- [[_COMMUNITY_Vector Index Config|Vector Index Config]]
- [[_COMMUNITY_HyDE Retrieval|HyDE Retrieval]]
- [[_COMMUNITY_Config Overlay & Health|Config Overlay & Health]]
- [[_COMMUNITY_Voice Transcription|Voice Transcription]]
- [[_COMMUNITY_Rate Limiter Class|Rate Limiter Class]]
- [[_COMMUNITY_Off-Topic Triage|Off-Topic Triage]]
- [[_COMMUNITY_App Configuration|App Configuration]]
- [[_COMMUNITY_WA Deduplication|WA Deduplication]]
- [[_COMMUNITY_Runtime Config Overlay|Runtime Config Overlay]]
- [[_COMMUNITY_Webhook Routes|Webhook Routes]]
- [[_COMMUNITY_Recall Evaluation Tests|Recall Evaluation Tests]]
- [[_COMMUNITY_Alembic Migrations|Alembic Migrations]]
- [[_COMMUNITY_WA Template Messages|WA Template Messages]]
- [[_COMMUNITY_Background Jobs|Background Jobs]]
- [[_COMMUNITY_Image Flush Operations|Image Flush Operations]]
- [[_COMMUNITY_Contextual Summarization|Contextual Summarization]]
- [[_COMMUNITY_Query Reformulation|Query Reformulation]]
- [[_COMMUNITY_Formatting Spec Tests|Formatting Spec Tests]]
- [[_COMMUNITY_Security Changelog|Security Changelog]]
- [[_COMMUNITY_LLM Fallback Chain|LLM Fallback Chain]]
- [[_COMMUNITY_Advanced Retrieval|Advanced Retrieval]]
- [[_COMMUNITY_Eval Fixtures|Eval Fixtures]]
- [[_COMMUNITY_Fallback Chain Parser|Fallback Chain Parser]]
- [[_COMMUNITY_Chunk Sibling Fetching|Chunk Sibling Fetching]]
- [[_COMMUNITY_Markdown Table Split|Markdown Table Split]]
- [[_COMMUNITY_Recall At 5 Eval|Recall At 5 Eval]]
- [[_COMMUNITY_Requirements & Integration|Requirements & Integration]]
- [[_COMMUNITY_Escalation Patterns|Escalation Patterns]]
- [[_COMMUNITY_History Summarization|History Summarization]]
- [[_COMMUNITY_Source Normalization|Source Normalization]]
- [[_COMMUNITY_Source Name Dots|Source Name Dots]]
- [[_COMMUNITY_Image Flush Test|Image Flush Test]]
- [[_COMMUNITY_Single Image Test|Single Image Test]]
- [[_COMMUNITY_Multi Image Test|Multi Image Test]]
- [[_COMMUNITY_Prefix Flush Test|Prefix Flush Test]]
- [[_COMMUNITY_Flush Task Safety|Flush Task Safety]]
- [[_COMMUNITY_Image Debounce Test|Image Debounce Test]]
- [[_COMMUNITY_Web Search RAG|Web Search RAG]]
- [[_COMMUNITY_WA Format Text|WA Format Text]]
- [[_COMMUNITY_Telegram Formatting Spec|Telegram Formatting Spec]]
- [[_COMMUNITY_WhatsApp Formatting Spec|WhatsApp Formatting Spec]]
- [[_COMMUNITY_RAG Tools Module|RAG Tools Module]]
- [[_COMMUNITY_Bot Handler Tests|Bot Handler Tests]]
- [[_COMMUNITY_Source Name Normalization|Source Name Normalization]]
- [[_COMMUNITY_Security Chunk Scan|Security Chunk Scan]]
- [[_COMMUNITY_Tool Use Call Tests|Tool Use Call Tests]]
- [[_COMMUNITY_Web Search Image Test|Web Search Image Test]]
- [[_COMMUNITY_WA Phone Normalization|WA Phone Normalization]]
- [[_COMMUNITY_WA Parse Incoming|WA Parse Incoming]]
- [[_COMMUNITY_WA Verify Webhook|WA Verify Webhook]]
- [[_COMMUNITY_WA Send Reply|WA Send Reply]]
- [[_COMMUNITY_Multi-Tenant SaaS Todos|Multi-Tenant SaaS Todos]]

## God Nodes (most connected - your core abstractions)
1. `rag_query()` - 64 edges
2. `WhatsAppAdapter` - 62 edges
3. `ChannelSendError` - 45 edges
4. `Tenant` - 43 edges
5. `ChannelButton` - 38 edges
6. `ChannelMessage` - 36 edges
7. `call_chat()` - 31 edges
8. `_make_ctx()` - 31 edges
9. `ImageBuffer` - 30 edges
10. `format_text_for_channel()` - 26 edges

## Surprising Connections (you probably didn't know these)
- `Single-Worker In-Process State Constraint` --rationale_for--> `Multi-Tenant Namespace Isolation`  [INFERRED]
  /root/telegram-bot/src/state.py → README.md
- `main()` --calls--> `AsyncOpenAI`  [INFERRED]
  scripts/test_mrl_dimensions.py → src/llm.py
- `Runtime Config Overlay (DB overrides .env)` --rationale_for--> `get_setting()`  [INFERRED]
  /root/telegram-bot/src/config_overlay.py → src/config_overlay.py
- `Multi-Tenant Isolation Pattern` --rationale_for--> `Tenant`  [INFERRED]
  /root/telegram-bot/src/state.py → src/db.py
- `test_index_chunks_skips_injected_chunk()` --calls--> `index_chunks()`  [INFERRED]
  tests/test_edge_cases.py → src/rag.py

## Import Cycles
- 1-file cycle: `src/lifespan.py -> src/lifespan.py`

## Hyperedges (group relationships)
- **MRL Embedding Dimension Migration Sequence (512→768 with HNSW rebuild)** — concept_mrl_embedding_dimension, versions_g7h8i9j0k1l2_mrl_vector_512_upgrade, versions_h8i9j0k1l2m3_mrl_vector_768_upgrade, scripts_test_mrl_dimensions_main, concept_hnsw_index_rebuild [EXTRACTED 0.95]
- **Linear Alembic Migration Chain (baseline→768-dim)** — versions_af07e4f0ea14_baseline_upgrade, versions_766462df7dd1_multi_tenant_schema_upgrade, versions_3f8a1c2e9d47_add_expertise_area_upgrade, versions_a1b2c3d4e5f6_smart_chatbot_v2_upgrade, versions_b2c3d4e5f6g7_whatsapp_multi_channel_upgrade, versions_c3d4e5f6g7h8_vision_web_search_upgrade, versions_d4e5f6g7h8i9_namespace_source_index_upgrade, versions_e5f6g7h8i9j0_system_config_and_indexes_upgrade, versions_f6g7h8i9j0k1_feedback_table_upgrade, versions_g7h8i9j0k1l2_mrl_vector_512_upgrade, versions_h8i9j0k1l2m3_mrl_vector_768_upgrade [EXTRACTED 1.00]
- **Tenant Provisioning Scripts (create, seed, test)** — scripts_create_tenant_main, scripts_seed_demo_seed, scripts_test_mrl_dimensions_main [INFERRED 0.75]
- **Channel Adapter Pattern: Protocol + TG + WA implementations** — channels_protocol_channeladapter, channels_telegram_telegramadapter, channels_whatsapp_whatsappadapter [EXTRACTED 0.95]
- **RAG Pipeline Core Flow: retrieve → generate → save** — src_rag_retrieve_context, src_rag_generate_answer, src_rag_save_turn, src_rag_rag_query [EXTRACTED 0.95]
- **Admin and API Document Upload Pipeline (normalize → parse → upsert → index)** — services_upload_normalize_source_name, services_upload_process_uploaded_file, concept_document_upsert, routes_admin_admin_upload_document, routes_api_upload_document [EXTRACTED 1.00]
- **Admin Tenant Lifecycle (create → init bot → toggle active → update → delete)** — routes_admin_admin_create_tenant, routes_admin_admin_update_tenant, routes_admin_admin_toggle_active, services_tenant_bot_init_tenant_bot [EXTRACTED 1.00]
- **Security Defense-in-Depth: sanitize input + scan chunks + canary token + validate output** — src_security_sanitize_user_input, src_security_scan_chunk_for_injection, src_security_canary_token, src_security_validate_output [EXTRACTED 0.95]
- **WA Message Processing Flow (webhook → adapter → service window → buffer → RAG → reply)** — routes_webhook_whatsapp_webhook, services_wa_processor_handle_wa_message, concept_wa_service_window, concept_image_buffer_wa, services_wa_processor_send_wa_reply [EXTRACTED 1.00]
- **Session-Scoped TestClient + DB Mock + Auth Override Pattern** — tests_conftest_app_client_fixture, tests_conftest_make_db_mock, tests_conftest_authed_api_client_fixture [EXTRACTED 1.00]
- **Vision Augmented Retrieval Test Coverage** — tests_test_edge_cases_vision_tests, tests_test_vision_generate_answer_image, concept_vision_augmented_retrieval [EXTRACTED 1.00]
- **N-Model LLM Fallback Chain Test Coverage** — tests_test_rag_pipeline_call_chat, concept_n_model_fallback_chain, changelog_v04220 [INFERRED 0.85]
- **Tenant Knowledge Base Document Pipeline** — concept_tenant_document_kb, documents_acme_fitness, documents_plantilla, sp_diagnostico_histologico [INFERRED 0.85]
- **Core RAG Architecture Concepts** — concept_rag_pipeline, concept_rag_guardrails, concept_llm_fallback, concept_pgvector_hnsw [INFERRED 0.90]
- **Infrastructure Deployment Modes (Dev vs Prod)** — docker_compose, concept_traefik_https, concept_ngrok_dev_tunnel, concept_single_worker_constraint [EXTRACTED 0.90]

## Communities (99 total, 27 thin omitted)

### Community 0 - "Telegram Bot Handlers"
Cohesion: 0.11
Nodes (50): DEFAULT_TYPE, cmd_clear(), cmd_contactar(), cmd_help(), cmd_sources(), cmd_start(), _get_tenant(), handle_message() (+42 more)

### Community 1 - "LLM Client Layer"
Cohesion: 0.06
Nodes (28): close_llm_clients(), _error_message(), extract_json_from_llm_response(), _mark_tool_use_failed(), Provider-agnostic LLM layer.  Supports any OpenAI-compatible chat/completion and, Send a tiny chat request to verify provider config.     Returns {"ok": True, "mo, Extract a JSON object from an LLM response.     Handles:     - Bare JSON: {"inte, Gracefully close HTTP clients. Call during app shutdown. (+20 more)

### Community 2 - "RAG Answer Generation"
Cohesion: 0.06
Nodes (45): RuntimeError, handle_photo(), _format_catalog_as_text(), generate_answer(), Format a complete price list from catalog chunks — no LLM required.      Parses, Generate an answer using retrieved context + conversation history.     When imag, When images is set, generate_answer must include an instruction     telling the, When low_confidence=True, generate_answer includes an approximate-match note (+37 more)

### Community 3 - "DB Models & Migrations"
Cohesion: 0.05
Nodes (19): PostgreSQL Database Backup 2026-06-03, Alembic Migration Chain (linear revision history), Document Upsert Pattern (delete+reinsert by namespace+source), Feedback Table (thumbs up/down ratings), System Config Runtime Overlay Table, Multi-Tenant DB Schema (tenants table), main(), One-time script to seed the first tenant. Run inside the api container:     dock (+11 more)

### Community 4 - "RAG Query Pipeline"
Cohesion: 0.06
Nodes (36): Dual Retrieval (HyDE + broad query), rag_query(), Full RAG pipeline: retrieve context → generate answer → save history.     Return, When images is set but LLM_VISION_MODEL is empty, rag_query returns     a clear, When images is set AND LLM_VISION_MODEL is configured, rag_query     proceeds no, When images is set but no text context found, rag_query sends the     image to t, When the vision model can't read the image, rag_query returns a     clear messag, When images sent with generic question and no context found, vision model     ex (+28 more)

### Community 5 - "Admin UI & Config"
Cohesion: 0.13
Nodes (34): Runtime Config Overlay via SystemConfig DB table, HTTP Basic Auth for Admin Panel, _admin_context(), admin_create_tenant(), admin_delete_docs(), admin_download_template(), admin_health_data(), admin_panel() (+26 more)

### Community 6 - "LLM Tool Use"
Cohesion: 0.07
Nodes (28): call_chat_with_tools(), is_tool_use_available(), Call chat/completions with tool_use support (OpenAI-compatible format).      If, _make_chat_response(), _make_error_response(), Tests for the native tool-use agent fan-out (T1–T5).  Covers all 22 code paths i, Build a mock httpx.Response for a chat/completions call., One tool raises an exception; the other succeeds; synthesis still runs. (+20 more)

### Community 7 - "History & Edge Case Tests"
Cohesion: 0.07
Nodes (21): get_history(), Edge-case test battery for the RAG bot.  Unit tests: no external deps, run anywh, Triage system prompt must prohibit self-introduction and greetings., Browser suffix (1) is stripped from filename., Browser suffix (2), (3) etc. are stripped., _copy and _copy2 suffixes are stripped., _2, _3 etc. before extension are stripped (Chrome download pattern)., Filenames are lowercased regardless of original casing. (+13 more)

### Community 8 - "Tenant & Base Models"
Cohesion: 0.13
Nodes (26): Base, HTTPBasicCredentials, Tenant, Conversation, DocumentChunk, get_db(), Database models and connection setup. Uses pgvector for similarity search on doc, Encrypted key-value store for system-level settings (LLM, embeddings, etc.). (+18 more)

### Community 9 - "Web Search Tests"
Cohesion: 0.07
Nodes (29): Tests for web search fallback: _web_search(), rag_query web search path, describ, When WEB_SEARCH_URL is empty, returns [] immediately (no HTTP call)., Results with content < 50 chars are filtered out., When no context found and tenant.web_search_enabled=True, falls back to web sear, Ollama-style response with 'results' key returns context chunks., When web_search_enabled=False, falls through to _triage_response., When web search fails (timeout), falls through to _triage_response., When context IS found in the KB, web search is never called. (+21 more)

### Community 10 - "Security & Input Guard"
Cohesion: 0.11
Nodes (27): Canary Token Prompt Injection Defense, CANARY_TOKEN, # NOTE: single-worker only — each uvicorn worker gets a different token., Normalize and check user input for injection patterns.      Returns the (possibl, Returns True if the chunk looks like an injection attempt.      Call inside inde, Log a warning if the LLM response contains the canary token.      Returns respon, sanitize_user_input(), scan_chunk_for_injection() (+19 more)

### Community 11 - "Channel Formatting"
Cohesion: 0.11
Nodes (8): CHANNEL_FORMATTING, format_text_for_channel(), Apply channel-specific post-processing to LLM output.      Universal normalizati, Telegram formatting — pass through (prompt already handles it)., Post-process LLM output for WhatsApp display rules., LLMs often emit **bold** but Telegram needs *bold*., Don't convert __ inside URLs or snake_case identifiers., TestFormatText

### Community 12 - "WhatsApp Adapter Tests"
Cohesion: 0.12
Nodes (9): make_wa_adapter(), make_wa_status_update(), make_wa_text_message(), Tests for WhatsApp adapter, channel-agnostic formatting, and service window logi, Second call with same msg_id returns empty list (dedup)., Fallback: verify_webhook reads from request._body when body param not given., TestParseIncoming, TestVerifyWebhook (+1 more)

### Community 13 - "Prompts & RAG Module"
Cohesion: 0.10
Nodes (25): System prompt builder for the RAG pipeline., _cache_key(), _dispatch_tool(), _get_cached(), _illegible_fallback_msg(), RAG Pipeline: Embed → Store → Retrieve → Answer  This is the core of the demo. S, Call the configured web search endpoint and return context chunks.     Tries Oll, Search indexed documents for a tool dispatch call.      Uses a fresh DB session (+17 more)

### Community 14 - "RAG Pipeline Tests"
Cohesion: 0.07
Nodes (16): _patch_lifespan_db(), Smoke tests for the RAG pipeline.  Unit tests run always (no external deps). Int, DELETE /namespace only removes the authenticated tenant's data., FAQ chunks (full_doc_text=None) must not call _add_contextual_summary., When full_doc_text is provided and llm_context_model is set, contextual summary, If _add_contextual_summary returns '', embed the chunk without context., DB must always store original chunk content, never the contextual text., Context manager that mocks DB calls in the lifespan (no tenants loaded). (+8 more)

### Community 15 - "WhatsApp Integration Tests"
Cohesion: 0.15
Nodes (9): make_tenant(), make_wa_adapter(), When >3 buttons, WA Cloud API requires <=3, so code falls back         to text-o, When outside 24h window with no template, _log_unanswered is called         inst, Test interactive list_reply (not just button_reply)., Create a mock Tenant with WA fields., TestParseIncomingMediaTypes, TestProcessWaMessage (+1 more)

### Community 16 - "LLM Call & Mock Tests"
Cohesion: 0.13
Nodes (24): call_chat(), Call chat/completions with primary model, then fallback on failure.     Returns, _mock_response(), Build a fake httpx.Response., Reasoning model returns content=null — should try next model, not return None., Reasoning model returns content=null with no fallback configured → RuntimeError., With 2 comma-separated fallbacks, 3rd model succeeds after primary+first fail., All 3 models in chain fail → RuntimeError. (+16 more)

### Community 17 - "Channel Message Types"
Cohesion: 0.13
Nodes (15): Any, Bot, ChannelMessage, Parsed message from any channel., Send reply via Telegram Bot API., Send 'typing' indicator via Telegram., Verify TG webhook using secret token comparison., TG doesn't use GET verification — returns None always. (+7 more)

### Community 18 - "Tenant Knowledge Docs"
Cohesion: 0.13
Nodes (23): Demo Tenant (Acme Fitness), Formalin-Only Sample Preservation Policy, Frozen Section (Corte Congelado) Special Handling Policy, Generic Knowledge Base Document Template, LLM Fallback Chain (Primary → Fallback Model), Multi-Tenant Namespace Isolation, Ngrok Dev Tunnel for Telegram Webhooks, PostgreSQL pgvector HNSW Index (+15 more)

### Community 19 - "WhatsApp Processor"
Cohesion: 0.14
Nodes (18): WA Image Accumulation Buffer with Flush Callback, Patch the Using Module Not the Defining Module, WhatsApp 24-Hour Service Window Table, create_wa_adapter(), handle_wa_message(), WhatsApp message processing — background task handler., Background task: process a single WA message through the RAG pipeline.      Crea, Send a WhatsApp reply with sources footer and escalation button.      Shared by (+10 more)

### Community 20 - "Crypto & Encryption"
Cohesion: 0.13
Nodes (15): decrypt_value(), encrypt_value(), generate_key(), get_fernet(), Symmetric encryption for secrets stored in the database.  Uses Fernet (AES-128-C, Return a lazy-initialised Fernet instance. Returns None if key not configured., Encrypt a string. Returns Fernet token (base64) or plaintext if key not set., Decrypt a Fernet token. Falls back to returning input as-is if decryption fails. (+7 more)

### Community 21 - "Test Infrastructure"
Cohesion: 0.10
Nodes (21): Session-Scoped TestClient Pattern, Tool Use Backoff Pattern, ToolUseNotSupportedError, api_client(), _app_client(), _app_client Session-Scoped TestClient Fixture, authed_api_client(), authed_api_client Fixture (authenticated) (+13 more)

### Community 22 - "Channel Adapter Protocol"
Cohesion: 0.11
Nodes (13): ChannelAdapter, Interface every messaging channel must implement., Parse raw webhook payload into ChannelMessage list.          Returns [] for non-, Download media from channel-specific URL. Returns raw bytes., Send reply. Buttons translated to channel-native format.         Returns API res, Send 'typing' indicator (best-effort)., Verify webhook signature. Returns True if valid., Handle channel verification (GET for WA hub.challenge).         Returns Response (+5 more)

### Community 23 - "WhatsApp Design Rationale"
Cohesion: 0.11
Nodes (9): Download media from WA Cloud API using media ID.          WA returns a media ID,, Send 'typing' indicator via WhatsApp (best-effort)., Verify WhatsApp webhook signature using HMAC-SHA256.          Callers should rea, Handle WA webhook GET verification (hub.challenge).          Returns Response if, ChannelAdapter implementation for WhatsApp Cloud API., WhatsAppAdapter, Request, Response (+1 more)

### Community 24 - "Document Chunking"
Cohesion: 0.11
Nodes (19): chunk_text(), Split text into semantically coherent chunks by splitting on paragraph     bound, Markdown table rows should become individual chunks with section headers,     no, Markdown table separator rows (|---|---|) should not appear in chunks., test_chunk_text_51_chars_included(), test_chunk_text_exactly_50_chars_skipped(), test_chunk_text_large_text_produces_multiple_chunks(), test_chunk_text_markdown_table_rows_become_separate_chunks() (+11 more)

### Community 25 - "Document Upload"
Cohesion: 0.17
Nodes (15): Idempotent Document Upsert (delete-then-insert pattern), admin_upload_document(), upload_document(), describe_image_for_upload(), detect_mime(), normalize_source_name(), process_uploaded_file(), File upload and vision processing — shared by API and admin routes. (+7 more)

### Community 26 - "Rate Limiting"
Cohesion: 0.11
Nodes (14): _api_key_func(), limiter (SlowAPI), RateLimitExceeded, rate_limit_handler(), Unified rate limiting — Telegram and WhatsApp.  Single RateLimiter class used by, Per-tenant rate limiting: use X-API-Key hash for authed API routes,     fall bac, FastAPI exception handler for slowapi rate limits., setup_logging() (+6 more)

### Community 27 - "Image Buffer Tests"
Cohesion: 0.11
Nodes (17): Tests for the image buffer module (multi-image debounce + flush)., Adding a second image cancels the first timer — only one flush happens., A single image is flushed after the buffer window expires., If on_flush raises, the entry is still removed from the buffer., flush_by_prefix flushes all entries whose key starts with the prefix., flush_by_prefix sets override_question on entries that have no question., When _flush is called from flush_by_prefix (not the timer task),     the pending, Adding a 6th image returns an error string; first 5 are kept. (+9 more)

### Community 28 - "Admin Edge Case Tests"
Cohesion: 0.16
Nodes (17): _admin_auth(), _make_db_mock(), Return (override_fn, mock_db) for use with app.dependency_overrides[get_db]., test_admin_create_tenant_duplicate_slug_shows_error(), test_admin_create_tenant_invalid_plan_shows_error(), test_admin_create_tenant_missing_slug_shows_error(), test_admin_panel_accessible_with_correct_credentials(), test_admin_upload_accepts_markdown() (+9 more)

### Community 29 - "Channel Buttons"
Cohesion: 0.16
Nodes (8): ChannelButton, Channel-agnostic button. Adapters translate to native format.      - url: opens, Tests for WhatsApp integration paths not covered by test_whatsapp_adapter.py.  C, TestChannelButton, TestDownloadMedia, TestFormatTextFallback, TestRagChannelParam, TestVerifyWebhookEdgeCases

### Community 30 - "System Prompt Builder"
Cohesion: 0.14
Nodes (12): Canary Token in System Prompt (prompt leak detection), Per-Channel Response Formatting (telegram vs whatsapp length/style), build_system_prompt(), _INJECTION_PATTERNS_ES (Spanish prompt injection detection patterns), Build the system prompt for the LLM, incorporating expertise area, channel forma, System prompt must instruct the LLM to provide near-match info instead     of sa, test_build_system_prompt_empty_area_no_trailing_dot_clause(), test_build_system_prompt_includes_expertise_area() (+4 more)

### Community 31 - "Multi-Tenant & Dev Tunnel"
Cohesion: 0.13
Nodes (12): Multi-Tenant Isolation Pattern, FastAPI, get_ngrok_domain(), Ngrok public URL discovery — shared by lifespan and admin., Query ngrok's local API (http://ngrok:4040) to get the public HTTPS URL.      Re, Telegram bot initialization — shared by lifespan and admin create-tenant., init_db(), FastAPI dependencies — shared across route modules. (+4 more)

### Community 32 - "REST API Routes"
Cohesion: 0.24
Nodes (14): BaseModel, delete_namespace(), get_feedback(), REST API routes — health, upload, stats, update_tenant, delete_namespace., Return recent feedback for the authenticated tenant's namespace., stats(), TenantUpdate, update_tenant() (+6 more)

### Community 33 - "Image Buffer"
Cohesion: 0.13
Nodes (11): ImageBuffer, For testing: number of pending buffer entries., Remove entries older than MAX_ENTRY_TTL. Returns count removed., For testing: clear all entries and cancel all tasks., In-memory image buffer with debounce flush.      add_image() appends an image an, Last non-empty question overwrites previous empty/blank questions., clear() cancels all pending flush tasks and removes entries., flush_by_prefix does NOT override if the entry already has a question. (+3 more)

### Community 34 - "Channel Protocol Types"
Cohesion: 0.19
Nodes (7): ChannelFormatting, normalize_phone(), Channel-agnostic protocol for multi-tenant messaging.  Every channel (Telegram,, Normalize a phone number for consistent DB lookups and rate limiting.      Strip, Per-channel formatting rules for LLM system prompts and output post-processing., Telegram channel adapter — Phase 1 scaffold.  Valid implementation of the Channe, TestNormalizePhone

### Community 35 - "Message Send & Errors"
Cohesion: 0.20
Nodes (7): ChannelSendError, Raised when a channel adapter fails to send a message., Send a text message via WhatsApp Cloud API.          Max 3 quick-reply buttons p, ChannelButton, Exception, TestChannelSendError, TestSendChatAction

### Community 36 - "WA Service Window"
Cohesion: 0.23
Nodes (8): check_wa_service_window(), WhatsApp Cloud API channel adapter.  Implements ChannelAdapter for Meta's WhatsA, Check if the 24-hour service window is still open for this user.      Returns Tr, UPSERT: update last_user_message_at for the 24h window tracker., update_wa_service_window(), WhatsApp 24-Hour Service Window Rule, WaServiceWindow, TestServiceWindow

### Community 37 - "Tenant State Registry"
Cohesion: 0.23
Nodes (9): init_tenant_bot(), Build telegram Application for a tenant and register its webhook.      Returns T, get_app(), Shared in-process state — telegram app registry.  Single-worker only (1 uvicorn, Refresh the cached Tenant object in an already-registered bot., register_app(), remove_app(), update_tenant_cache() (+1 more)

### Community 38 - "Image Debounce Changelog"
Cohesion: 0.18
Nodes (12): Changelog v0.4.0.0 — Vision-Augmented Retrieval, Changelog v0.4.1.0 — Vision + Multi-image Buffer, Image Buffer Debounce Pattern, Low-Confidence Retrieval Fallback (0.10–0.20 similarity), Vision-Augmented Retrieval Pipeline, chunk_text Edge Case Tests, Vision Pipeline Edge Case Tests, ImageBuffer Test Suite (+4 more)

### Community 39 - "Image Buffer Entries"
Cohesion: 0.18
Nodes (9): BufferedImage, _BufferEntry, Image buffer — collects multiple images before RAG processing.  Single-worker on, For testing: inspect a buffer entry., A single base64-encoded image awaiting processing., Internal buffer state for one grouping key., Add an image to the buffer.          Returns None on success, or an error messag, sweep() removes entries older than MAX_ENTRY_TTL. (+1 more)

### Community 40 - "Vision Search Terms"
Cohesion: 0.17
Nodes (12): _extract_search_terms_from_images(), Use the vision model to extract key search terms from images.      When a user s, _extract_search_terms_from_images calls vision model and returns terms., _extract_search_terms_from_images returns empty string on LLM failure., _extract_search_terms_from_images returns "" when no vision model configured., _extract_search_terms_from_images returns "" when call_chat returns empty string, _extract_search_terms_from_images builds content with multiple image_url parts., test_extract_search_terms_empty_string_result() (+4 more)

### Community 41 - "Response Validation"
Cohesion: 0.17
Nodes (12): _is_illegible_response(), Check if the vision model's response indicates it couldn't read the image.     M, _is_illegible_response detects Spanish illegibility phrases., _is_illegible_response detects English illegibility phrases., _is_illegible_response does not flag normal vision responses., Partially legible responses should NOT trigger the full-illegible fallback., Full illegibility phrases without partial qualifiers still trigger., test_is_illegible_response_allows_normal_response() (+4 more)

### Community 42 - "Embeddings Client"
Cohesion: 0.18
Nodes (11): AsyncOpenAI, call_embeddings(), _get_embedding_client(), Embed a list of texts via the configured embedding provider.     Returns list of, Get or create the embedding OpenAI client., test_call_embeddings_api_error_raises_runtime_error(), test_call_embeddings_empty_list_returns_empty(), test_call_embeddings_rate_limit_raises_runtime_error() (+3 more)

### Community 43 - "Vector Index Config"
Cohesion: 0.25
Nodes (6): HNSW Index Rebuild Pattern (drop→alter→recreate), MRL (Matryoshka Representation Learning) Embedding Dimension, main(), Test whether the configured embedding provider supports MRL (Matryoshka) dimensi, upgrade(), upgrade()

### Community 44 - "HyDE Retrieval"
Cohesion: 0.18
Nodes (11): HyDE (Hypothetical Document Embeddings), _hyde_query(), Generate a hypothetical catalog/document answer and return it as the search key., Short but valid hypotheticals like 'Biopsia.' (8 chars) must be accepted., Responses shorter than 3 chars are noise — reject them., test_hyde_query_2_chars_rejected(), test_hyde_query_8_chars_accepted(), test_hyde_query_empty_response_returns_empty() (+3 more)

### Community 45 - "Config Overlay & Health"
Cohesion: 0.20
Nodes (4): health(), get_setting(), Return DB override if present, else fallback (.env value).      Note: `or` seman, TestConfigOverlay

### Community 46 - "Voice Transcription"
Cohesion: 0.22
Nodes (7): Speech-to-text — Groq Whisper transcription., Transcribe audio bytes using Groq Whisper. Raises RuntimeError on failure., transcribe_voice(), TestSttAdditional, test_transcribe_voice_429(), test_transcribe_voice_success(), test_transcribe_voice_timeout()

### Community 47 - "Rate Limiter Class"
Cohesion: 0.18
Nodes (8): RateLimiter, Generic sliding-window rate limiter.      check(key) returns True if the user is, Return True if key is rate-limited (>= max_messages in window)., Remove entries whose window has fully expired. Returns count removed., test_per_user_rate_limit_burst(), test_per_user_rate_limit_independent_users(), test_per_user_rate_limit_window_rollover(), test_rate_limit_dict_cleanup_after_window_expires()

### Community 48 - "Off-Topic Triage"
Cohesion: 0.20
Nodes (10): Classify intent and generate fallback reply when no context found.     Returns (, _triage_response(), que planes tienes?' must be classified as ambiguous, not greeting., Pure social 'hi' should stay as greeting intent., test_triage_ambiguous_classified_for_plans_question(), test_triage_greeting_only_for_pure_social(), test_triage_response_invalid_json_returns_fallback(), test_triage_response_network_failure_returns_fallback() (+2 more)

### Community 49 - "App Configuration"
Cohesion: 0.22
Nodes (5): BaseSettings, Embedding API key. Falls back to openrouter_api_key for backwards compat., LLM_API_KEY takes precedence over openrouter_api_key., Fallback API key resolution:         1. LLM_FALLBACK_API_KEY if explicitly set, Settings

### Community 50 - "WA Deduplication"
Cohesion: 0.28
Nodes (5): _is_duplicate(), Check if a WA message ID was already processed. Thread-safe for single worker., Parse WA webhook payload into ChannelMessage list.          Returns [] for statu, ChannelMessage, TestMessageDedup

### Community 51 - "Runtime Config Overlay"
Cohesion: 0.22
Nodes (7): Runtime Config Overlay (DB overrides .env), get_setting_int(), AsyncSession, Runtime config overlay: DB-stored settings override .env values without restart., Return DB override cast to int if present, else fallback., Load all SystemConfig rows into the in-process overlay.     Per-row decrypt erro, reload_from_db()

### Community 52 - "Webhook Routes"
Cohesion: 0.33
Nodes (8): Webhook routes — Telegram and WhatsApp., WhatsApp webhook GET verification (hub.challenge)., WhatsApp webhook POST — receive and process messages.      Sync path: tenant loo, telegram_webhook(), whatsapp_webhook(), whatsapp_webhook_verify(), AsyncSession, Request

### Community 53 - "Recall Evaluation Tests"
Cohesion: 0.36
Nodes (7): _chunk_matches(), _is_placeholder(), _load_fixture(), Eval harness: recall@5 for the vector search pipeline.  Measures how often the e, Return True if a retrieved chunk satisfies the eval pair criteria., recall@5 = fraction of labeled pairs where the expected chunk     appears in the, test_recall_at_5()

### Community 54 - "Alembic Migrations"
Cohesion: 0.48
Nodes (6): do_run_migrations(), get_url(), run_async_migrations(), run_migrations_offline(), run_migrations_online(), Connection

### Community 55 - "WA Template Messages"
Cohesion: 0.47
Nodes (3): Send a pre-approved template message outside the 24h window.      Returns API re, send_wa_template(), TestSendWaTemplate

### Community 56 - "Background Jobs"
Cohesion: 0.40
Nodes (5): cleanup_job(), daily_digest_job(), Background jobs — daily digest and weekly cleanup., Send top unanswered queries to each tenant's operator via Telegram., Weekly: purge old UnansweredQuery rows and sweep stale rate-limit entries.

### Community 57 - "Image Flush Operations"
Cohesion: 0.33
Nodes (3): Sleep for delay, then flush the buffer entry., Process buffered images for the given key., Flush all entries whose key starts with prefix.          Used when a text/voice

### Community 58 - "Contextual Summarization"
Cohesion: 0.33
Nodes (6): _add_contextual_summary(), Generate a 1-2 sentence context summary to prepend before embedding.      The su, When LLM_CONTEXT_MODEL is not configured, skip contextual summary., LLM failure must return '' and not raise — upload must not fail., test_add_contextual_summary_llm_failure_returns_empty(), test_add_contextual_summary_no_model_returns_empty()

### Community 59 - "Query Reformulation"
Cohesion: 0.33
Nodes (6): Rewrite a follow-up question into a standalone query using conversation history., _reformulate_query(), _reformulate_query returns the original question when history is empty., _reformulate_query calls the LLM to rewrite follow-up questions., test_reformulate_query_calls_llm_when_history_present(), test_reformulate_query_returns_original_when_no_history()

### Community 61 - "Security Changelog"
Cohesion: 0.40
Nodes (5): Changelog v0.3.0.0 — Runtime Config Overlay, Canary Token Exfiltration Protection, History Sanitization Tests, sanitize_user_input Tests, validate_output Tests

### Community 62 - "LLM Fallback Chain"
Cohesion: 0.50
Nodes (4): Changelog v0.4.2.0 — N-model Fallback Chain, N-Model LLM Fallback Chain with Per-Model Timeout, call_chat Fallback Chain Tests, TODOS — LLM Fallback Chain Backlog

### Community 63 - "Advanced Retrieval"
Cohesion: 0.50
Nodes (4): Contextual Retrieval — LLM Summary Prepended to Chunk Embedding, HyDE (Hypothetical Document Embedding) Query, Contextual Retrieval / index_chunks Tests, HyDE Query Tests

### Community 64 - "Eval Fixtures"
Cohesion: 0.50
Nodes (3): _instructions, pairs, threshold

### Community 65 - "Fallback Chain Parser"
Cohesion: 0.50
Nodes (4): _parse_fallback_chain(), Parse comma-separated model names, stripping whitespace and empty entries., _parse_fallback_chain strips whitespace, trailing commas, and empty entries., test_parse_fallback_chain_helper()

### Community 66 - "Chunk Sibling Fetching"
Cohesion: 0.50
Nodes (4): _chunk_base_term(), _fetch_section_siblings(), Extract the base/category term from a table-row chunk's item name.      Chunk fo, Fetch sibling catalog rows from the same section as retrieved chunks.      When

### Community 67 - "Markdown Table Split"
Cohesion: 0.50
Nodes (4): Pre-process markdown tables: separate each row into its own paragraph     so chu, _split_markdown_tables(), _split_markdown_tables should prepend the section header to each table row., test_split_markdown_tables_prepends_header()

### Community 68 - "Recall At 5 Eval"
Cohesion: 0.67
Nodes (3): recall@5 Evaluation Metric for Vector Search, Eval Pairs Fixture JSON, recall@5 Eval Harness

## Ambiguous Edges - Review These
- `telegram_webhook()` → `create_wa_adapter()`  [AMBIGUOUS]
  /root/telegram-bot/src/routes/webhook.py · relation: calls

## Knowledge Gaps
- **74 isolated node(s):** `Connection`, `Response`, `AsyncSession`, `RateLimitExceeded`, `Request` (+69 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **27 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **What is the exact relationship between `telegram_webhook()` and `create_wa_adapter()`?**
  _Edge tagged AMBIGUOUS (relation: calls) - confidence is low._
- **Why does `rag_query()` connect `RAG Query Pipeline` to `Telegram Bot Handlers`, `LLM Client Layer`, `RAG Answer Generation`, `LLM Tool Use`, `History & Edge Case Tests`, `Tenant & Base Models`, `Web Search Tests`, `Security & Input Guard`, `Prompts & RAG Module`, `LLM Call & Mock Tests`, `Tenant Knowledge Docs`, `WhatsApp Processor`, `System Prompt Builder`, `Vision Search Terms`, `Response Validation`, `HyDE Retrieval`, `Config Overlay & Health`, `Off-Topic Triage`, `Query Reformulation`, `Chunk Sibling Fetching`?**
  _High betweenness centrality (0.174) - this node is a cross-community bridge._
- **Why does `Tenant` connect `Tenant & Base Models` to `Telegram Bot Handlers`, `REST API Routes`, `LLM Client Layer`, `Admin UI & Config`, `Tenant State Registry`, `Config Overlay & Health`, `Voice Transcription`, `WhatsApp Processor`, `Crypto & Encryption`, `Webhook Routes`, `Document Upload`, `Rate Limiting`, `Multi-Tenant & Dev Tunnel`?**
  _High betweenness centrality (0.120) - this node is a cross-community bridge._
- **Why does `WhatsAppAdapter` connect `WhatsApp Design Rationale` to `LLM Client Layer`, `Tenant & Base Models`, `Channel Formatting`, `WhatsApp Adapter Tests`, `WhatsApp Integration Tests`, `Channel Message Types`, `WhatsApp Processor`, `Crypto & Encryption`, `Channel Adapter Protocol`, `Document Upload`, `Rate Limiting`, `Channel Buttons`, `System Prompt Builder`, `Channel Protocol Types`, `Message Send & Errors`, `WA Service Window`, `Tenant State Registry`, `Config Overlay & Health`, `Voice Transcription`, `WA Deduplication`, `WA Template Messages`, `Formatting Spec Tests`?**
  _High betweenness centrality (0.108) - this node is a cross-community bridge._
- **Are the 37 inferred relationships involving `rag_query()` (e.g. with `Dual Retrieval (HyDE + broad query)` and `RAG Pipeline Flow`) actually correct?**
  _`rag_query()` has 37 INFERRED edges - model-reasoned connections that need verification._
- **Are the 44 inferred relationships involving `WhatsAppAdapter` (e.g. with `TelegramAdapter` and `ChannelButton`) actually correct?**
  _`WhatsAppAdapter` has 44 INFERRED edges - model-reasoned connections that need verification._
- **Are the 35 inferred relationships involving `ChannelSendError` (e.g. with `Any` and `Bot`) actually correct?**
  _`ChannelSendError` has 35 INFERRED edges - model-reasoned connections that need verification._