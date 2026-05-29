"""
Tests for WhatsApp adapter, channel-agnostic formatting, and service window logic.

Covers:
- WhatsAppAdapter.parse_incoming (text, voice, image, status, button, interactive, malformed)
- WhatsAppAdapter.verify_webhook (HMAC-SHA256)
- WhatsAppAdapter.handle_verification (GET hub.challenge)
- WhatsAppAdapter.format_text (table stripping, backtick stripping, truncation)
- normalize_phone (E.164, +prefix, whatsapp: prefix)
- format_text_for_channel (Telegram passthrough, WhatsApp post-processing)
- _build_system_prompt channel parameter
- ChannelFormatting specs
- Message dedup
"""

import hashlib
import hmac
import os
import sys
import time
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

# Ensure src/ is on the path so imports like "from channels.protocol import ..." work
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest

from channels.protocol import (
    CHANNEL_FORMATTING,
    ChannelButton,
    ChannelMessage,
    ChannelSendError,
    TELEGRAM_FORMATTING,
    WHATSAPP_FORMATTING,
    format_text_for_channel,
    normalize_phone,
)
from channels.whatsapp import (
    WhatsAppAdapter,
    _is_duplicate,
    _wa_message_cache,
)


# ─── Fixtures ──────────────────────────────────────────────────────────────

def make_wa_adapter():
    return WhatsAppAdapter(
        phone_number_id="1234567890",
        access_token="test_token",
        app_secret="test_secret",
        verify_token="test_verify",
        business_id="biz_123",
    )


def make_wa_text_message(text="Hola", from_number="549111234567", msg_id="wamid_abc123"):
    return {
        "object": "whatsapp_business_account",
        "entry": [{
            "id": "biz_123",
            "changes": [{
                "value": {
                    "messaging_product": "whatsapp",
                    "messages": [{
                        "from": from_number,
                        "id": msg_id,
                        "timestamp": "1700000000",
                        "type": "text",
                        "text": {"body": text},
                    }],
                },
            }],
        }],
    }


def make_wa_status_update():
    return {
        "object": "whatsapp_business_account",
        "entry": [{
            "id": "biz_123",
            "changes": [{
                "value": {
                    "statuses": [{"id": "wamid_xyz", "status": "delivered", "timestamp": "1700000000"}],
                },
            }],
        }],
    }


# ─── normalize_phone ──────────────────────────────────────────────────────

class TestNormalizePhone:
    def test_strips_plus_prefix(self):
        assert normalize_phone("+549111234567") == "549111234567"

    def test_strips_whatsapp_prefix(self):
        assert normalize_phone("whatsapp:+549111234567") == "549111234567"

    def test_already_clean(self):
        assert normalize_phone("549111234567") == "549111234567"

    def test_whitespace_stripped(self):
        assert normalize_phone("  +549111234567  ") == "549111234567"

    def test_whatsapp_without_plus(self):
        assert normalize_phone("whatsapp:549111234567") == "549111234567"


# ─── parse_incoming ────────────────────────────────────────────────────────

class TestParseIncoming:
    @pytest.mark.asyncio
    async def test_text_message(self):
        adapter = make_wa_adapter()
        msgs = await adapter.parse_incoming(make_wa_text_message("Hola", "549111234567", "wamid_1"))
        assert len(msgs) == 1
        assert msgs[0].text == "Hola"
        assert msgs[0].user_id == "549111234567"
        assert msgs[0].channel == "whatsapp"

    @pytest.mark.asyncio
    async def test_voice_message(self):
        payload = {
            "object": "whatsapp_business_account",
            "entry": [{
                "id": "biz_123",
                "changes": [{
                    "value": {
                        "messaging_product": "whatsapp",
                        "messages": [{
                            "from": "549111234567",
                            "id": "wamid_voice1",
                            "type": "audio",
                            "audio": {"id": "audio_123", "mime_type": "audio/ogg"},
                        }],
                    },
                }],
            }],
        }
        adapter = make_wa_adapter()
        msgs = await adapter.parse_incoming(payload)
        assert len(msgs) == 1
        assert msgs[0].media_type == "voice"
        assert msgs[0].media_url == "audio_123"

    @pytest.mark.asyncio
    async def test_image_message(self):
        payload = {
            "object": "whatsapp_business_account",
            "entry": [{
                "id": "biz_123",
                "changes": [{
                    "value": {
                        "messaging_product": "whatsapp",
                        "messages": [{
                            "from": "549111234567",
                            "id": "wamid_img1",
                            "type": "image",
                            "image": {"id": "img_123"},
                        }],
                    },
                }],
            }],
        }
        adapter = make_wa_adapter()
        msgs = await adapter.parse_incoming(payload)
        assert len(msgs) == 1
        assert msgs[0].media_type == "image"
        assert msgs[0].text is None

    @pytest.mark.asyncio
    async def test_status_callback_returns_empty(self):
        adapter = make_wa_adapter()
        msgs = await adapter.parse_incoming(make_wa_status_update())
        assert msgs == []

    @pytest.mark.asyncio
    async def test_button_callback(self):
        payload = {
            "object": "whatsapp_business_account",
            "entry": [{
                "id": "biz_123",
                "changes": [{
                    "value": {
                        "messaging_product": "whatsapp",
                        "messages": [{
                            "from": "549111234567",
                            "id": "wamid_btn1",
                            "type": "button",
                            "button": {"text": "Ayuda", "payload": "help"},
                        }],
                    },
                }],
            }],
        }
        adapter = make_wa_adapter()
        msgs = await adapter.parse_incoming(payload)
        assert len(msgs) == 1
        assert msgs[0].text == "Ayuda"
        assert msgs[0].reply_to == "help"

    @pytest.mark.asyncio
    async def test_interactive_reply(self):
        payload = {
            "object": "whatsapp_business_account",
            "entry": [{
                "id": "biz_123",
                "changes": [{
                    "value": {
                        "messaging_product": "whatsapp",
                        "messages": [{
                            "from": "549111234567",
                            "id": "wamid_int1",
                            "type": "interactive",
                            "interactive": {
                                "button_reply": {"id": "sources_btn", "title": "Ver fuentes"},
                            },
                        }],
                    },
                }],
            }],
        }
        adapter = make_wa_adapter()
        msgs = await adapter.parse_incoming(payload)
        assert len(msgs) == 1
        assert msgs[0].text == "Ver fuentes"
        assert msgs[0].reply_to == "sources_btn"

    @pytest.mark.asyncio
    async def test_malformed_payload_raises(self):
        adapter = make_wa_adapter()
        with pytest.raises(ValueError, match="Invalid"):
            await adapter.parse_incoming({})

    @pytest.mark.asyncio
    async def test_unsupported_msg_type_returns_empty(self):
        payload = {
            "object": "whatsapp_business_account",
            "entry": [{
                "id": "biz_123",
                "changes": [{
                    "value": {
                        "messaging_product": "whatsapp",
                        "messages": [{
                            "from": "549111234567",
                            "id": "wamid_unsup1",
                            "type": "reaction",
                            "reaction": {"emoji": "\U0001f44d"},
                        }],
                    },
                }],
            }],
        }
        adapter = make_wa_adapter()
        msgs = await adapter.parse_incoming(payload)
        assert msgs == []

    @pytest.mark.asyncio
    async def test_dedup_skips_duplicate_message(self):
        """Second call with same msg_id returns empty list (dedup)."""
        adapter = make_wa_adapter()
        _wa_message_cache.clear()
        payload = make_wa_text_message("Hola", "549111234567", "wamid_dedup1")
        msgs1 = await adapter.parse_incoming(payload)
        assert len(msgs1) == 1
        msgs2 = await adapter.parse_incoming(payload)
        assert len(msgs2) == 0  # Deduped


# ─── verify_webhook ────────────────────────────────────────────────────────

class TestVerifyWebhook:
    def test_valid_signature_with_body_param(self):
        adapter = make_wa_adapter()
        body = b'{"entry":[]}'
        sig = "sha256=" + hmac.new(b"test_secret", body, hashlib.sha256).hexdigest()
        request = MagicMock()
        request.headers = {"X-Hub-Signature-256": sig}
        assert adapter.verify_webhook(request, body) is True

    def test_valid_signature_with_request_body(self):
        """Fallback: verify_webhook reads from request._body when body param not given."""
        adapter = make_wa_adapter()
        body = b'{"entry":[]}'
        sig = "sha256=" + hmac.new(b"test_secret", body, hashlib.sha256).hexdigest()
        request = MagicMock()
        request.headers = {"X-Hub-Signature-256": sig}
        request._body = body
        assert adapter.verify_webhook(request) is True

    def test_invalid_signature(self):
        adapter = make_wa_adapter()
        request = MagicMock()
        request.headers = {"X-Hub-Signature-256": "sha256=invalid"}
        request._body = b'{"entry":[]}'
        assert adapter.verify_webhook(request) is False

    def test_missing_signature(self):
        adapter = make_wa_adapter()
        request = MagicMock()
        request.headers = {}
        request._body = b'{"entry":[]}'
        assert adapter.verify_webhook(request) is False

    def test_empty_body_returns_false(self):
        adapter = make_wa_adapter()
        request = MagicMock()
        request.headers = {"X-Hub-Signature-256": "sha256=somesig"}
        # No body param, no _body attribute
        assert adapter.verify_webhook(request) is False


# ─── handle_verification ──────────────────────────────────────────────────

class TestHandleVerification:
    @pytest.mark.asyncio
    async def test_valid_challenge(self):
        adapter = make_wa_adapter()
        request = MagicMock()
        request.query_params = {"hub.mode": "subscribe", "hub.verify_token": "test_verify", "hub.challenge": "challenge_token_123"}
        response = await adapter.handle_verification(request)
        assert response is not None
        assert response.body == b"challenge_token_123"

    @pytest.mark.asyncio
    async def test_wrong_verify_token(self):
        adapter = make_wa_adapter()
        request = MagicMock()
        request.query_params = {"hub.mode": "subscribe", "hub.verify_token": "wrong_token", "hub.challenge": "challenge_123"}
        response = await adapter.handle_verification(request)
        assert response is None


# ─── format_text_for_channel ──────────────────────────────────────────────

class TestFormatText:
    def test_telegram_passthrough(self):
        text = "Hola *negrita* y `código`"
        assert format_text_for_channel(text, "telegram") == text

    def test_whatsapp_strips_tables(self):
        text = "Precio:\n| Plan | Precio |\n| Basic | $29 |\n*Plan Basic* — $29"
        result = format_text_for_channel(text, "whatsapp")
        assert "|" not in result
        assert "*Plan Basic*" in result

    def test_whatsapp_strips_backticks(self):
        text = "El precio es `$29` por mes"
        result = format_text_for_channel(text, "whatsapp")
        assert "`" not in result
        assert "$29" in result

    def test_whatsapp_truncation(self):
        text = "A" * 5000
        result = format_text_for_channel(text, "whatsapp")
        assert len(result) <= 4096
        assert result.endswith("...")

    def test_telegram_no_truncation_under_limit(self):
        text = "A" * 3000
        result = format_text_for_channel(text, "telegram")
        assert len(result) == 3000  # Under 4096, no truncation

    def test_whatsapp_collapse_blank_lines(self):
        text = "Line1\n\n\n\nLine2"
        result = format_text_for_channel(text, "whatsapp")
        assert "\n\n\n" not in result
        assert "Line1" in result
        assert "Line2" in result

    def test_whatsapp_strips_markdown_headers(self):
        text = "## Precios\n\n*Plan Basic* — $29"
        result = format_text_for_channel(text, "whatsapp")
        assert "##" not in result
        assert "*Precios*" in result

    def test_whatsapp_converts_strikethrough(self):
        text = "Precio ~~$50~~ *$29*"
        result = format_text_for_channel(text, "whatsapp")
        assert "~~" not in result
        assert "~$50~" in result
        assert "*$29*" in result

    def test_whatsapp_strips_horizontal_rules(self):
        text = "Intro\n---\nPrecio"
        result = format_text_for_channel(text, "whatsapp")
        assert "---" not in result
        assert "Intro" in result
        assert "Precio" in result

    def test_whatsapp_strips_code_block_language_hint(self):
        text = "```python\nprint('hello')\n```"
        result = format_text_for_channel(text, "whatsapp")
        assert "```python" not in result
        assert "```\n" in result


# ─── _build_system_prompt channel param ────────────────────────────────────

class TestBuildSystemPromptChannel:
    def test_telegram_formatting(self):
        from rag import _build_system_prompt
        prompt = _build_system_prompt("test_area", channel="telegram")
        assert "Formato para Telegram" in prompt
        assert "backticks" in prompt

    def test_whatsapp_formatting(self):
        from rag import _build_system_prompt
        prompt = _build_system_prompt("test_area", channel="whatsapp")
        assert "Formato para WhatsApp" in prompt
        assert "NUNCA uses `backticks" in prompt
        assert "```" in prompt  # triple-backtick monospace is supported
        assert "~tachado~" in prompt  # strikethrough format

    def test_default_is_telegram(self):
        from rag import _build_system_prompt
        prompt_default = _build_system_prompt("test_area")
        prompt_tg = _build_system_prompt("test_area", channel="telegram")
        assert prompt_default == prompt_tg


# ─── Message dedup ────────────────────────────────────────────────────────

class TestMessageDedup:
    def test_first_message_not_duplicate(self):
        _wa_message_cache.clear()
        assert _is_duplicate("msg_001") is False

    def test_same_message_is_duplicate(self):
        _wa_message_cache.clear()
        _is_duplicate("msg_002")
        assert _is_duplicate("msg_002") is True

    def test_different_messages_not_duplicate(self):
        _wa_message_cache.clear()
        _is_duplicate("msg_003")
        assert _is_duplicate("msg_004") is False


# ─── ChannelFormatting specs ──────────────────────────────────────────────

class TestChannelFormattingSpecs:
    def test_telegram_spec_exists(self):
        assert "telegram" in CHANNEL_FORMATTING
        fmt = CHANNEL_FORMATTING["telegram"]
        assert fmt.max_length == 4096
        assert fmt.supports_markdown_tables is False

    def test_whatsapp_spec_exists(self):
        assert "whatsapp" in CHANNEL_FORMATTING
        fmt = CHANNEL_FORMATTING["whatsapp"]
        assert fmt.max_length == 4096
        assert fmt.code_format == "text"

    def test_unknown_channel_defaults_to_telegram(self):
        fmt = CHANNEL_FORMATTING.get("unknown_channel", CHANNEL_FORMATTING["telegram"])
        assert fmt.max_length == 4096

    def test_telegram_allows_backticks(self):
        assert TELEGRAM_FORMATTING.code_format == "`text`"

    def test_whatsapp_forbids_backticks(self):
        assert WHATSAPP_FORMATTING.code_format == "text"


# ─── WhatsAppAdapter.format_text ──────────────────────────────────────────

class TestWhatsAppAdapterFormatText:
    def test_delegates_to_format_text_for_channel(self):
        adapter = make_wa_adapter()
        text = "Hello `code` world"
        result = adapter.format_text(text)
        assert "`" not in result
        assert "code" in result

    def test_truncates_long_text(self):
        adapter = make_wa_adapter()
        text = "X" * 5000
        result = adapter.format_text(text)
        assert len(result) <= 4096


# ─── ChannelSendError ────────────────────────────────────────────────────

class TestChannelSendError:
    def test_error_message(self):
        err = ChannelSendError("whatsapp", "549111234567", RuntimeError("timeout"))
        assert "[whatsapp]" in str(err)
        assert "549111234567" in str(err)
        assert err.channel == "whatsapp"
        assert err.original_error is not None

    def test_is_exception(self):
        err = ChannelSendError("telegram", "123", ValueError("test"))
        assert isinstance(err, Exception)


# ─── ChannelMessage defaults ──────────────────────────────────────────────

class TestChannelMessage:
    def test_default_channel_is_telegram(self):
        msg = ChannelMessage(user_id="123")
        assert msg.channel == "telegram"

    def test_text_defaults_none(self):
        msg = ChannelMessage(user_id="123")
        assert msg.text is None

    def test_raw_defaults_empty_dict(self):
        msg = ChannelMessage(user_id="123")
        assert msg.raw == {}