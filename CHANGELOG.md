# Changelog

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