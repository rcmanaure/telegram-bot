"""
Tests for WhatsApp integration paths not covered by test_whatsapp_adapter.py.

Covers:
- WhatsAppAdapter.send_reply (text, buttons, error, >3 buttons fallback)
- WhatsAppAdapter.send_chat_action
- WhatsAppAdapter.download_media (success, error)
- WhatsAppAdapter.verify_webhook with empty body / no body attribute
- check_wa_service_window / update_wa_service_window (with mocked DB)
- send_wa_template (success, empty template, error)
- format_text_for_channel unknown channel fallback, custom max_length
- ChannelButton dataclass
- _get_wa_adapter helper
- _process_wa_message branches (media fallback, voice, sanitize, rate limit,
  outside-window paths, RAG error)
- rag.py channel param propagation
- parse_incoming: document/video/sticker/list_reply branches
"""

import hashlib
import hmac
import json
import sys
import os
import time
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
import httpx

from channels.protocol import (
    CHANNEL_FORMATTING,
    ChannelButton,
    ChannelMessage,
    ChannelSendError,
    format_text_for_channel,
    normalize_phone,
)
from channels.whatsapp import (
    WhatsAppAdapter,
    _is_duplicate,
    _wa_message_cache,
    check_wa_service_window,
    update_wa_service_window,
    send_wa_template,
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


def make_tenant(**overrides):
    """Create a mock Tenant with WA fields."""
    t = MagicMock()
    t.id = 1
    t.slug = "test-tenant"
    t.expertise_area = "fitness"
    t.wa_phone_number_id = overrides.get("wa_phone_number_id", "1234567890")
    t.wa_access_token = overrides.get("wa_access_token", "test_token")
    t.wa_app_secret = overrides.get("wa_app_secret", "test_secret")
    t.wa_verify_token = overrides.get("wa_verify_token", "test_verify")
    t.wa_business_id = overrides.get("wa_business_id", "biz_123")
    t.wa_reengagement_template = overrides.get("wa_reengagement_template", None)
    t.channels = overrides.get("channels", "telegram,whatsapp")
    return t


# ─── ChannelButton ──────────────────────────────────────────────────────────

class TestChannelButton:
    def test_button_defaults(self):
        b = ChannelButton(label="Click me")
        assert b.label == "Click me"
        assert b.url is None
        assert b.callback_data is None

    def test_button_with_url(self):
        b = ChannelButton(label="Visit", url="https://example.com")
        assert b.url == "https://example.com"

    def test_button_with_callback(self):
        b = ChannelButton(label="Go", callback_data="go_payload")
        assert b.callback_data == "go_payload"


# ─── format_text_for_channel: unknown channel fallback ─────────────────────

class TestFormatTextFallback:
    def test_unknown_channel_defaults_to_telegram(self):
        text = "Hello `code` world"
        result = format_text_for_channel(text, "unknown_channel")
        # Should behave like telegram (passthrough)
        assert result == text

    def test_unknown_channel_truncation(self):
        text = "A" * 5000
        result = format_text_for_channel(text, "unknown_channel")
        assert len(result) <= 4096

    def test_custom_max_length(self):
        text = "B" * 200
        result = format_text_for_channel(text, "telegram", max_length=100)
        # text[:100-3] + "..." = 97 + 3 = 100
        assert len(result) == 100
        assert result.endswith("...")


# ─── WhatsAppAdapter.send_reply ────────────────────────────────────────────

class TestSendReply:
    @pytest.mark.asyncio
    async def test_send_text_only(self):
        adapter = make_wa_adapter()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"messages": [{"id": "wamid_abc"}]}

        with patch.object(adapter._http, "post", new_callable=AsyncMock, return_value=mock_resp):
            result = await adapter.send_reply("549111234567", "Hello")
        assert result["messages"][0]["id"] == "wamid_abc"

    @pytest.mark.asyncio
    async def test_send_with_buttons(self):
        adapter = make_wa_adapter()
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"messages": [{"id": "wamid_btn"}]}

        buttons = [
            ChannelButton(label="Yes", callback_data="yes"),
            ChannelButton(label="No", callback_data="no"),
        ]

        with patch.object(adapter._http, "post", new_callable=AsyncMock, return_value=mock_resp) as mock_post:
            result = await adapter.send_reply("549111234567", "Choose", buttons=buttons)

        call_args = mock_post.call_args
        payload = call_args.kwargs.get("json", call_args[1].get("json"))
        assert payload["type"] == "interactive"
        assert len(payload["interactive"]["action"]["buttons"]) == 2

    @pytest.mark.asyncio
    async def test_send_reply_more_than_3_buttons_falls_back_to_text(self):
        """When >3 buttons, WA Cloud API requires <=3, so code falls back
        to text-only (no interactive payload)."""
        adapter = make_wa_adapter()
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"messages": [{"id": "wamid_text"}]}

        buttons = [
            ChannelButton(label=f"Btn{i}", callback_data=f"btn_{i}")
            for i in range(5)
        ]

        with patch.object(adapter._http, "post", new_callable=AsyncMock, return_value=mock_resp) as mock_post:
            await adapter.send_reply("549111234567", "Pick", buttons=buttons)

        call_args = mock_post.call_args
        payload = call_args.kwargs.get("json", call_args[1].get("json"))
        # When >3 buttons, code does not set interactive type — stays as text
        assert payload["type"] == "text"
        assert "text" in payload

    @pytest.mark.asyncio
    async def test_send_reply_http_error_raises_channel_send_error(self):
        adapter = make_wa_adapter()

        with patch.object(adapter._http, "post", new_callable=AsyncMock,
                         side_effect=httpx.HTTPStatusError(
                             "error", request=MagicMock(), response=MagicMock(status_code=401))):
            with pytest.raises(ChannelSendError) as exc_info:
                await adapter.send_reply("549111234567", "Hello")

        assert exc_info.value.channel == "whatsapp"
        assert exc_info.value.user_id == "549111234567"


# ─── WhatsAppAdapter.send_chat_action ──────────────────────────────────────

class TestSendChatAction:
    @pytest.mark.asyncio
    async def test_send_chat_action_best_effort(self):
        adapter = make_wa_adapter()
        # Should not raise even on error
        with patch.object(adapter._http, "post", new_callable=AsyncMock,
                         side_effect=Exception("network error")):
            await adapter.send_chat_action("549111234567")  # No exception

    @pytest.mark.asyncio
    async def test_send_chat_action_success(self):
        adapter = make_wa_adapter()
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()

        with patch.object(adapter._http, "post", new_callable=AsyncMock, return_value=mock_resp) as mock_post:
            await adapter.send_chat_action("549111234567")

        assert mock_post.called


# ─── WhatsAppAdapter.download_media ────────────────────────────────────────

class TestDownloadMedia:
    @pytest.mark.asyncio
    async def test_download_media_success(self):
        adapter = make_wa_adapter()

        url_resp = MagicMock()
        url_resp.json.return_value = {"url": "https://example.com/media.ogg"}
        url_resp.raise_for_status = MagicMock()

        content_resp = MagicMock()
        content_resp.content = b"audio-bytes"
        content_resp.raise_for_status = MagicMock()

        with patch.object(adapter._http, "get", new_callable=AsyncMock,
                         side_effect=[url_resp, content_resp]) as mock_get:
            result = await adapter.download_media("media_123")

        assert result == b"audio-bytes"
        assert mock_get.call_count == 2

    @pytest.mark.asyncio
    async def test_download_media_no_url_raises(self):
        adapter = make_wa_adapter()

        url_resp = MagicMock()
        url_resp.json.return_value = {}  # No "url" key
        url_resp.raise_for_status = MagicMock()

        with patch.object(adapter._http, "get", new_callable=AsyncMock, return_value=url_resp):
            with pytest.raises(ChannelSendError) as exc_info:
                await adapter.download_media("media_456")

        assert exc_info.value.channel == "whatsapp"


# ─── WhatsAppAdapter.verify_webhook: empty body ───────────────────────────

class TestVerifyWebhookEdgeCases:
    def test_empty_body_returns_false(self):
        adapter = make_wa_adapter()
        request = MagicMock()
        request.headers = {"X-Hub-Signature-256": "sha256=abc"}
        request._body = b""  # empty body
        result = adapter.verify_webhook(request)
        assert result is False

    def test_no_body_attribute_returns_false(self):
        adapter = make_wa_adapter()
        request = MagicMock(spec=[])  # no attributes
        request.headers = {}
        result = adapter.verify_webhook(request)
        assert result is False


# ─── check_wa_service_window / update_wa_service_window ────────────────────

class TestServiceWindow:
    @pytest.mark.asyncio
    async def test_check_window_no_record_returns_false(self):
        mock_db = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_db.execute = AsyncMock(return_value=mock_result)

        result = await check_wa_service_window(mock_db, tenant_id=1, user_id="549111234567")
        assert result is False

    @pytest.mark.asyncio
    async def test_check_window_within_24h_returns_true(self):
        from db import WaServiceWindow

        mock_db = AsyncMock()
        mock_window = MagicMock(spec=WaServiceWindow)
        mock_window.last_user_message_at = datetime.now(timezone.utc) - timedelta(hours=1)
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_window
        mock_db.execute = AsyncMock(return_value=mock_result)

        result = await check_wa_service_window(mock_db, tenant_id=1, user_id="549111234567")
        assert result is True

    @pytest.mark.asyncio
    async def test_check_window_outside_24h_returns_false(self):
        from db import WaServiceWindow

        mock_db = AsyncMock()
        mock_window = MagicMock(spec=WaServiceWindow)
        mock_window.last_user_message_at = datetime.now(timezone.utc) - timedelta(hours=25)
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_window
        mock_db.execute = AsyncMock(return_value=mock_result)

        result = await check_wa_service_window(mock_db, tenant_id=1, user_id="549111234567")
        assert result is False

    @pytest.mark.asyncio
    async def test_update_window_creates_new_record(self):
        mock_db = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_db.execute = AsyncMock(return_value=mock_result)

        await update_wa_service_window(mock_db, tenant_id=1, user_id="549111234567")
        assert mock_db.add.called
        assert mock_db.commit.called

    @pytest.mark.asyncio
    async def test_update_window_updates_existing_record(self):
        from db import WaServiceWindow

        mock_window = MagicMock(spec=WaServiceWindow)
        mock_db = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_window
        mock_db.execute = AsyncMock(return_value=mock_result)

        await update_wa_service_window(mock_db, tenant_id=1, user_id="549111234567")
        assert mock_window.last_user_message_at is not None
        assert mock_db.commit.called


# ─── send_wa_template ───────────────────────────────────────────────────────

class TestSendWaTemplate:
    @pytest.mark.asyncio
    async def test_send_template_success(self):
        adapter = make_wa_adapter()
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"messages": [{"id": "wamid_tpl"}]}

        with patch.object(adapter._http, "post", new_callable=AsyncMock, return_value=mock_resp) as mock_post:
            result = await send_wa_template(adapter, "549111234567", "hello_template")

        assert result is not None
        assert result["messages"][0]["id"] == "wamid_tpl"
        # Verify payload has template type
        call_args = mock_post.call_args
        payload = call_args.kwargs.get("json", call_args[1].get("json"))
        assert payload["type"] == "template"

    @pytest.mark.asyncio
    async def test_send_template_empty_name_returns_none(self):
        adapter = make_wa_adapter()
        result = await send_wa_template(adapter, "549111234567", "")
        assert result is None

    @pytest.mark.asyncio
    async def test_send_template_http_error_raises(self):
        adapter = make_wa_adapter()

        with patch.object(adapter._http, "post", new_callable=AsyncMock,
                         side_effect=httpx.HTTPStatusError(
                             "error", request=MagicMock(), response=MagicMock(status_code=403))):
            with pytest.raises(ChannelSendError):
                await send_wa_template(adapter, "549111234567", "hello_template")


# ─── create_wa_adapter helper ────────────────────────────────────────────────

class TestGetWaAdapter:
    def test_returns_adapter_when_wa_configured(self):
        from services.wa_processor import create_wa_adapter
        tenant = make_tenant()
        adapter = create_wa_adapter(tenant)
        assert adapter is not None
        assert isinstance(adapter, WhatsAppAdapter)
        assert adapter.phone_number_id == "1234567890"

    def test_returns_none_when_wa_not_configured(self):
        from services.wa_processor import create_wa_adapter
        tenant = make_tenant(wa_phone_number_id=None, wa_access_token=None)
        adapter = create_wa_adapter(tenant)
        assert adapter is None

    def test_returns_none_when_channels_excludes_whatsapp(self):
        from services.wa_processor import create_wa_adapter
        tenant = make_tenant(channels="telegram")
        adapter = create_wa_adapter(tenant)
        assert adapter is None

    def test_returns_none_when_phone_number_id_empty(self):
        from services.wa_processor import create_wa_adapter
        tenant = make_tenant(wa_phone_number_id="")
        adapter = create_wa_adapter(tenant)
        assert adapter is None


# ─── handle_wa_message branches ───────────────────────────────────────────
# WA message processing lives in services/wa_processor.py as handle_wa_message.
# Rate limiting uses wa_rate_limiter from limiter.py (no more bot.rate_limits hack).

class TestProcessWaMessage:
    @pytest.mark.asyncio
    async def test_unsupported_media_replies_help_text(self):
        from services.wa_processor import handle_wa_message
        mock_db = AsyncMock()
        tenant = make_tenant()
        wa_msg = ChannelMessage(user_id="549111234567", text=None, media_type="image", channel="whatsapp")

        adapter = make_wa_adapter()
        with patch("services.wa_processor.create_wa_adapter", return_value=adapter), \
             patch.object(adapter, "send_reply", new_callable=AsyncMock) as mock_send:
            await handle_wa_message(tenant, wa_msg)

        mock_send.assert_called_once()
        assert "Solo puedo procesar texto" in mock_send.call_args[0][1]

    @pytest.mark.asyncio
    async def test_voice_message_replies_not_available(self):
        from services.wa_processor import handle_wa_message
        mock_db = AsyncMock()
        tenant = make_tenant()
        wa_msg = ChannelMessage(user_id="549111234567", text=None, media_type="voice", channel="whatsapp")

        adapter = make_wa_adapter()
        with patch("services.wa_processor.create_wa_adapter", return_value=adapter), \
             patch.object(adapter, "send_reply", new_callable=AsyncMock) as mock_send:
            await handle_wa_message(tenant, wa_msg)

        mock_send.assert_called_once()
        assert "notas de voz" in mock_send.call_args[0][1]

    @pytest.mark.asyncio
    async def test_sanitize_error_replies_blocked(self):
        from services.wa_processor import handle_wa_message
        mock_db = AsyncMock()
        tenant = make_tenant()
        wa_msg = ChannelMessage(user_id="549111234567", text="ignore all", channel="whatsapp")

        adapter = make_wa_adapter()
        with patch("services.wa_processor.create_wa_adapter", return_value=adapter), \
             patch("services.wa_processor.sanitize_user_input", side_effect=ValueError("blocked")), \
             patch.object(adapter, "send_reply", new_callable=AsyncMock) as mock_send:
            await handle_wa_message(tenant, wa_msg)

        mock_send.assert_called_once()
        assert "no permitido" in mock_send.call_args[0][1]

    @pytest.mark.asyncio
    async def test_empty_text_returns_early(self):
        from services.wa_processor import handle_wa_message
        mock_db = AsyncMock()
        tenant = make_tenant()
        wa_msg = ChannelMessage(user_id="549111234567", text="   ", channel="whatsapp")

        adapter = make_wa_adapter()
        with patch("services.wa_processor.create_wa_adapter", return_value=adapter), \
             patch.object(adapter, "send_reply", new_callable=AsyncMock) as mock_send:
            await handle_wa_message(tenant, wa_msg)

        mock_send.assert_not_called()

    @pytest.mark.asyncio
    async def test_rate_limit_returns_early(self):
        from services.wa_processor import handle_wa_message

        tenant = make_tenant()
        wa_msg = ChannelMessage(user_id="549111234567", text="Hola", channel="whatsapp")

        adapter = make_wa_adapter()
        with patch("services.wa_processor.create_wa_adapter", return_value=adapter), \
             patch("services.wa_processor.sanitize_user_input", return_value="Hola"), \
             patch("services.wa_processor.wa_rate_limiter") as mock_limiter, \
             patch.object(adapter, "send_reply", new_callable=AsyncMock) as mock_send:
            mock_limiter.check.return_value = True  # rate limited
            await handle_wa_message(tenant, wa_msg)
        mock_send.assert_not_called()

    @pytest.mark.asyncio
    async def test_outside_window_with_template_sends_template(self):
        from services.wa_processor import handle_wa_message
        tenant = make_tenant(wa_reengagement_template="hello_template")

        wa_msg = ChannelMessage(user_id="549111234567", text="Hola", channel="whatsapp")

        adapter = make_wa_adapter()
        with patch("services.wa_processor.create_wa_adapter", return_value=adapter), \
             patch("services.wa_processor.sanitize_user_input", return_value="Hola"), \
             patch("services.wa_processor.check_wa_service_window", new_callable=AsyncMock, return_value=False), \
             patch("services.wa_processor.send_wa_template", new_callable=AsyncMock) as mock_tpl:
            await handle_wa_message(tenant, wa_msg)

        mock_tpl.assert_called_once()

    @pytest.mark.asyncio
    async def test_outside_window_no_template_logs_unanswered(self):
        """When outside 24h window with no template, _log_unanswered is called
        instead of crashing with NameError (fixed in commit 8f0d6e1)."""
        from services.wa_processor import handle_wa_message
        tenant = make_tenant(wa_reengagement_template=None)

        wa_msg = ChannelMessage(user_id="549111234567", text="Hola", channel="whatsapp")

        adapter = make_wa_adapter()
        with patch("services.wa_processor.create_wa_adapter", return_value=adapter), \
             patch("services.wa_processor.sanitize_user_input", return_value="Hola"), \
             patch("services.wa_processor.check_wa_service_window", new_callable=AsyncMock, return_value=False), \
             patch("services.wa_processor._log_unanswered", new_callable=AsyncMock) as mock_log:
            await handle_wa_message(tenant, wa_msg)

        mock_log.assert_called_once()

    @pytest.mark.asyncio
    async def test_rag_error_sends_error_message(self):
        from services.wa_processor import handle_wa_message
        tenant = make_tenant()
        wa_msg = ChannelMessage(user_id="549111234567", text="Hola", channel="whatsapp")

        adapter = make_wa_adapter()
        with patch("services.wa_processor.create_wa_adapter", return_value=adapter), \
             patch("services.wa_processor.sanitize_user_input", return_value="Hola"), \
             patch("services.wa_processor.check_wa_service_window", new_callable=AsyncMock, return_value=True), \
             patch("services.wa_processor.update_wa_service_window", new_callable=AsyncMock), \
             patch("services.wa_processor.rag_query", new_callable=AsyncMock, side_effect=RuntimeError("LLM down")), \
             patch.object(adapter, "send_reply", new_callable=AsyncMock) as mock_send:
            await handle_wa_message(tenant, wa_msg)

        mock_send.assert_called_once()
        assert "error" in mock_send.call_args[0][1].lower() or "Lo siento" in mock_send.call_args[0][1]


# ─── rag.py channel param propagation ──────────────────────────────────────

class TestRagChannelParam:
    def test_generate_answer_default_channel(self):
        """generate_answer should accept channel param without error."""
        from services.prompts import build_system_prompt
        prompt_tg = build_system_prompt("test", channel="telegram")
        prompt_wa = build_system_prompt("test", channel="whatsapp")
        assert prompt_tg != prompt_wa
        assert "Telegram" in prompt_tg
        assert "WhatsApp" in prompt_wa


# ─── parse_incoming: document/video/sticker branch ─────────────────────────

class TestParseIncomingMediaTypes:
    @pytest.mark.asyncio
    async def test_document_message(self):
        payload = {
            "object": "whatsapp_business_account",
            "entry": [{
                "id": "biz_123",
                "changes": [{
                    "value": {
                        "messaging_product": "whatsapp",
                        "messages": [{
                            "from": "549111234567",
                            "id": "wamid_doc1",
                            "type": "document",
                            "document": {"id": "doc_123"},
                        }],
                    },
                }],
            }],
        }
        adapter = make_wa_adapter()
        _wa_message_cache.clear()
        msgs = await adapter.parse_incoming(payload)
        assert len(msgs) == 1
        assert msgs[0].media_type == "image"  # unsupported media maps to "image"

    @pytest.mark.asyncio
    async def test_video_message(self):
        payload = {
            "object": "whatsapp_business_account",
            "entry": [{
                "id": "biz_123",
                "changes": [{
                    "value": {
                        "messaging_product": "whatsapp",
                        "messages": [{
                            "from": "549111234567",
                            "id": "wamid_vid1",
                            "type": "video",
                            "video": {"id": "vid_123"},
                        }],
                    },
                }],
            }],
        }
        adapter = make_wa_adapter()
        _wa_message_cache.clear()
        msgs = await adapter.parse_incoming(payload)
        assert len(msgs) == 1
        assert msgs[0].media_type == "image"

    @pytest.mark.asyncio
    async def test_sticker_message(self):
        payload = {
            "object": "whatsapp_business_account",
            "entry": [{
                "id": "biz_123",
                "changes": [{
                    "value": {
                        "messaging_product": "whatsapp",
                        "messages": [{
                            "from": "549111234567",
                            "id": "wamid_stk1",
                            "type": "sticker",
                            "sticker": {"id": "stk_123"},
                        }],
                    },
                }],
            }],
        }
        adapter = make_wa_adapter()
        _wa_message_cache.clear()
        msgs = await adapter.parse_incoming(payload)
        assert len(msgs) == 1
        assert msgs[0].media_type == "image"

    @pytest.mark.asyncio
    async def test_interactive_list_reply(self):
        """Test interactive list_reply (not just button_reply)."""
        payload = {
            "object": "whatsapp_business_account",
            "entry": [{
                "id": "biz_123",
                "changes": [{
                    "value": {
                        "messaging_product": "whatsapp",
                        "messages": [{
                            "from": "549111234567",
                            "id": "wamid_list1",
                            "type": "interactive",
                            "interactive": {
                                "list_reply": {"id": "option_1", "title": "Option One"},
                            },
                        }],
                    },
                }],
            }],
        }
        adapter = make_wa_adapter()
        _wa_message_cache.clear()
        msgs = await adapter.parse_incoming(payload)
        assert len(msgs) == 1
        assert msgs[0].text == "Option One"
        assert msgs[0].reply_to == "option_1"