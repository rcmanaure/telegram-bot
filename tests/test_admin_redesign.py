"""
Tests for feat/admin-redesign codepaths not covered by existing test files.

Covers:
- config_overlay: get_setting, get_setting_int, reload_from_db
- crypto: get_fernet, encrypt_value, decrypt_value, generate_key
- llm: extract_json_from_llm_response, test_connection, _get_embedding_client, _error_message
- state: register_app, get_app, update_tenant_cache, remove_app
- routes/webhook: whatsapp_webhook POST, whatsapp_webhook_verify GET
- routes/admin: admin_update_tenant, admin_queries, contact_url validation
- services/upload: detect_mime, describe_image_for_upload
- services/stt: transcribe_voice 401 and generic error
- rag: sync_faq_chunks, save_turn, _log_unanswered direct
- limiter: rate_limit_handler
"""
import base64
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
import httpx


# ─── config_overlay ────────────────────────────────────────────────────────────

class TestConfigOverlay:
    def test_get_setting_returns_fallback_when_no_overlay(self):
        from config_overlay import get_setting
        # Default overlay is empty
        assert get_setting("nonexistent_key", "fallback_val") == "fallback_val"

    def test_get_setting_returns_overlay_when_present(self):
        import config_overlay
        config_overlay._overlay["test_key"] = "overlay_val"
        try:
            assert config_overlay.get_setting("test_key", "fallback_val") == "overlay_val"
        finally:
            config_overlay._overlay.clear()

    def test_get_setting_returns_empty_string_default(self):
        from config_overlay import get_setting
        assert get_setting("absent_key") == ""

    def test_get_setting_int_returns_int_overlay(self):
        import config_overlay
        config_overlay._overlay["max_msgs"] = "30"
        try:
            assert config_overlay.get_setting_int("max_msgs", 20) == 30
        finally:
            config_overlay._overlay.clear()

    def test_get_setting_int_returns_fallback_on_bad_value(self):
        import config_overlay
        config_overlay._overlay["max_msgs"] = "not_a_number"
        try:
            assert config_overlay.get_setting_int("max_msgs", 20) == 20
        finally:
            config_overlay._overlay.clear()

    def test_get_setting_int_returns_fallback_when_absent(self):
        from config_overlay import get_setting_int
        assert get_setting_int("absent_key", 42) == 42

    @pytest.mark.asyncio
    async def test_reload_from_db_populates_overlay(self):
        import config_overlay
        config_overlay._overlay.clear()

        mock_row = MagicMock()
        mock_row.key = "llm_api_key"
        mock_row.encrypted_value = "plaintext_value"

        mock_result = MagicMock()
        mock_result.fetchall.return_value = [mock_row]

        mock_db = AsyncMock()
        mock_db.execute = AsyncMock(return_value=mock_result)

        # Patch decrypt_value at the module it's imported from (crypto.decrypt_value)
        with patch("crypto.decrypt_value", return_value="plaintext_value"):
            await config_overlay.reload_from_db(mock_db)

        assert config_overlay._overlay["llm_api_key"] == "plaintext_value"
        config_overlay._overlay.clear()

    @pytest.mark.asyncio
    async def test_reload_from_db_handles_decrypt_failure(self):
        import config_overlay
        config_overlay._overlay.clear()

        good_row = MagicMock()
        good_row.key = "good_key"
        good_row.encrypted_value = "good_val"

        bad_row = MagicMock()
        bad_row.key = "bad_key"
        bad_row.encrypted_value = "corrupt_val"

        mock_result = MagicMock()
        mock_result.fetchall.return_value = [good_row, bad_row]

        mock_db = AsyncMock()
        mock_db.execute = AsyncMock(return_value=mock_result)

        def side_effect_decrypt(val):
            if val == "corrupt_val":
                raise ValueError("decrypt failed")
            return val

        with patch("crypto.decrypt_value", side_effect=side_effect_decrypt):
            await config_overlay.reload_from_db(mock_db)

        assert "good_key" in config_overlay._overlay
        assert "bad_key" not in config_overlay._overlay
        config_overlay._overlay.clear()

    @pytest.mark.asyncio
    async def test_reload_from_db_handles_db_error(self):
        import config_overlay
        config_overlay._overlay.clear()

        mock_db = AsyncMock()
        mock_db.execute = AsyncMock(side_effect=Exception("DB connection lost"))

        # Should not raise, just log warning
        await config_overlay.reload_from_db(mock_db)
        assert len(config_overlay._overlay) == 0


# ─── crypto ────────────────────────────────────────────────────────────────────

class TestCrypto:
    def test_encrypt_decrypt_roundtrip(self):
        try:
            from cryptography.fernet import Fernet
        except ImportError:
            pytest.skip("cryptography package not installed")

        from crypto import encrypt_value, decrypt_value, generate_key
        real_key = generate_key()
        with patch.dict(os.environ, {"ENCRYPTION_KEY": real_key}):
            import crypto
            crypto._fernet = None
            crypto._warned_missing_key = False
            encrypted = encrypt_value("hello world")
            assert encrypted != "hello world"
            decrypted = decrypt_value(encrypted)
            assert decrypted == "hello world"
            crypto._fernet = None

    def test_encrypt_value_empty_string(self):
        from crypto import encrypt_value
        with patch.dict(os.environ, {"ENCRYPTION_KEY": ""}, clear=False):
            import crypto
            crypto._fernet = None
            crypto._warned_missing_key = False
            assert encrypt_value("") == ""
            crypto._fernet = None

    def test_decrypt_value_plaintext_fallback(self):
        from crypto import decrypt_value
        with patch.dict(os.environ, {"ENCRYPTION_KEY": ""}, clear=False):
            import crypto
            crypto._fernet = None
            crypto._warned_missing_key = False
            # No key → values pass through as plaintext
            assert decrypt_value("plain_text") == "plain_text"
            crypto._fernet = None

    def test_get_fernet_returns_none_when_no_key(self):
        from crypto import get_fernet
        with patch.dict(os.environ, {"ENCRYPTION_KEY": ""}, clear=False):
            import crypto
            crypto._fernet = None
            crypto._warned_missing_key = False
            assert get_fernet() is None
            crypto._fernet = None

    def test_decrypt_value_corrupt_ciphertext_returns_plaintext(self):
        """When encryption key is set but value isn't a valid Fernet token,
        decrypt_value falls back to returning the input as-is."""
        try:
            from cryptography.fernet import Fernet
        except ImportError:
            pytest.skip("cryptography package not installed")

        from crypto import decrypt_value
        # Use a valid Fernet key
        with patch.dict(os.environ, {"ENCRYPTION_KEY": Fernet.generate_key().decode()}):
            import crypto
            crypto._fernet = None
            crypto._warned_missing_key = False
            # "not-a-fernet-token" should be returned as-is (backward compat)
            result = decrypt_value("not-a-fernet-token")
            assert result == "not-a-fernet-token"
            crypto._fernet = None


# ─── llm.extract_json_from_llm_response ───────────────────────────────────────

class TestExtractJson:
    def test_bare_json(self):
        from llm import extract_json_from_llm_response
        result = extract_json_from_llm_response('{"intent": "greeting", "reply": "Hola"}')
        assert result == {"intent": "greeting", "reply": "Hola"}

    def test_markdown_wrapped_json(self):
        from llm import extract_json_from_llm_response
        text = '```json\n{"intent": "off_topic", "reply": "No"}\n```'
        result = extract_json_from_llm_response(text)
        assert result == {"intent": "off_topic", "reply": "No"}

    def test_json_with_surrounding_text(self):
        from llm import extract_json_from_llm_response
        text = 'Sure! Here is the result: {"intent": "ambiguous", "reply": "Try again"}'
        result = extract_json_from_llm_response(text)
        assert result["intent"] == "ambiguous"

    def test_no_json_raises_value_error(self):
        from llm import extract_json_from_llm_response
        with pytest.raises(ValueError, match="No JSON object found"):
            extract_json_from_llm_response("just plain text no json at all")

    def test_markdown_block_no_json_label(self):
        from llm import extract_json_from_llm_response
        text = '```\n{"intent": "needs_human", "reply": "Contactanos"}\n```'
        result = extract_json_from_llm_response(text)
        assert result["intent"] == "needs_human"


# ─── llm.test_connection ──────────────────────────────────────────────────────

class TestLlmTestConnection:
    @pytest.mark.asyncio
    async def test_connection_success(self):
        from llm import test_connection
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"choices": [{"message": {"content": "ok"}}]}

        with patch("llm.httpx.AsyncClient") as MockClient:
            mock_instance = AsyncMock()
            mock_instance.post = AsyncMock(return_value=mock_resp)
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = mock_instance

            result = await test_connection("https://api.test.com/v1", "key", "model-x")

        assert result["ok"] is True
        assert result["model"] == "model-x"

    @pytest.mark.asyncio
    async def test_connection_connect_error(self):
        from llm import test_connection
        with patch("llm.httpx.AsyncClient") as MockClient:
            mock_instance = AsyncMock()
            mock_instance.post = AsyncMock(side_effect=httpx.ConnectError("refused"))
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = mock_instance

            result = await test_connection("https://bad.url", "key", "m")

        assert result["ok"] is False
        assert "Cannot connect" in result["error"]

    @pytest.mark.asyncio
    async def test_connection_timeout(self):
        from llm import test_connection
        with patch("llm.httpx.AsyncClient") as MockClient:
            mock_instance = AsyncMock()
            mock_instance.post = AsyncMock(side_effect=httpx.TimeoutException("timeout"))
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = mock_instance

            result = await test_connection("https://slow.url", "key", "m")

        assert result["ok"] is False
        assert "timed out" in result["error"]


# ─── llm._error_message ──────────────────────────────────────────────────────

class TestLlmErrorMessage:
    def test_402_payment_required(self):
        from llm import _error_message
        resp = MagicMock()
        resp.status_code = 402
        resp.headers = {}
        resp.text = "Payment Required"
        err = httpx.HTTPStatusError("402", request=MagicMock(), response=resp)
        msg = _error_message(err)
        assert "payment required" in msg.lower()

    def test_429_rate_limited(self):
        from llm import _error_message
        resp = MagicMock()
        resp.status_code = 429
        resp.headers = {}
        resp.text = "Too Many Requests"
        err = httpx.HTTPStatusError("429", request=MagicMock(), response=resp)
        msg = _error_message(err)
        assert "rate-limited" in msg.lower()

    def test_timeout_exception(self):
        from llm import _error_message
        err = httpx.TimeoutException("timeout")
        msg = _error_message(err)
        assert "timed out" in msg.lower()


# ─── state module ─────────────────────────────────────────────────────────────

class TestStateModule:
    def test_register_and_get_app(self):
        from state import register_app, get_app, telegram_apps
        token = "test_token_state_123"
        mock_app = MagicMock()
        register_app(token, mock_app)
        try:
            assert get_app(token) is mock_app
        finally:
            telegram_apps.pop(token, None)

    def test_get_app_returns_none_for_unknown(self):
        from state import get_app
        assert get_app("nonexistent_token_xyz") is None

    def test_update_tenant_cache(self):
        from state import register_app, update_tenant_cache, telegram_apps
        token = "test_token_update_456"
        mock_app = MagicMock()
        mock_app.bot_data = {}
        register_app(token, mock_app)
        try:
            tenant = MagicMock()
            tenant.bot_token = token
            update_tenant_cache(tenant)
            assert mock_app.bot_data["tenant"] is tenant
        finally:
            telegram_apps.pop(token, None)

    def test_remove_app(self):
        from state import register_app, remove_app, get_app, telegram_apps
        token = "test_token_remove_789"
        mock_app = MagicMock()
        register_app(token, mock_app)
        assert get_app(token) is not None
        remove_app(token)
        assert get_app(token) is None


# ─── webhook routes: WhatsApp ──────────────────────────────────────────────────

class TestWhatsappWebhookRoutes:
    def test_wa_webhook_verify_success(self, api_client):
        import main as main_module
        from db import get_db, Tenant
        from channels.whatsapp import WhatsAppAdapter

        mock_tenant = MagicMock(spec=Tenant)
        mock_tenant.slug = "wa-tenant"
        mock_tenant.active = True
        mock_tenant.channels = "telegram,whatsapp"
        mock_tenant.wa_phone_number_id = "123"
        mock_tenant.wa_access_token = "tok"
        mock_tenant.wa_app_secret = "secret"
        mock_tenant.wa_verify_token = "verify123"
        mock_tenant.wa_business_id = "biz"

        db_override, _ = _make_db_mock(scalar_one_or_none=mock_tenant)
        main_module.app.dependency_overrides[get_db] = db_override
        try:
            r = api_client.get(
                "/webhook/wa-tenant/whatsapp?hub.mode=subscribe&hub.verify_token=verify123&hub.challenge=challenge_xyz"
            )
            assert r.status_code == 200
            assert r.content == b"challenge_xyz"
        finally:
            main_module.app.dependency_overrides.pop(get_db, None)

    def test_wa_webhook_verify_wrong_token(self, api_client):
        import main as main_module
        from db import get_db, Tenant

        mock_tenant = MagicMock(spec=Tenant)
        mock_tenant.slug = "wa-tenant"
        mock_tenant.active = True
        mock_tenant.channels = "telegram,whatsapp"
        mock_tenant.wa_phone_number_id = "123"
        mock_tenant.wa_access_token = "tok"
        mock_tenant.wa_app_secret = "secret"
        mock_tenant.wa_verify_token = "verify123"
        mock_tenant.wa_business_id = "biz"

        db_override, _ = _make_db_mock(scalar_one_or_none=mock_tenant)
        main_module.app.dependency_overrides[get_db] = db_override
        try:
            r = api_client.get(
                "/webhook/wa-tenant/whatsapp?hub.mode=subscribe&hub.verify_token=wrong&hub.challenge=xyz"
            )
            assert r.status_code == 403
        finally:
            main_module.app.dependency_overrides.pop(get_db, None)

    def test_wa_webhook_post_hmac_failure(self, api_client):
        import main as main_module
        from db import get_db, Tenant

        mock_tenant = MagicMock(spec=Tenant)
        mock_tenant.slug = "wa-tenant"
        mock_tenant.active = True
        mock_tenant.channels = "telegram,whatsapp"
        mock_tenant.wa_phone_number_id = "123"
        mock_tenant.wa_access_token = "tok"
        mock_tenant.wa_app_secret = "secret"
        mock_tenant.wa_verify_token = "verify123"
        mock_tenant.wa_business_id = "biz"

        db_override, _ = _make_db_mock(scalar_one_or_none=mock_tenant)
        main_module.app.dependency_overrides[get_db] = db_override
        try:
            r = api_client.post(
                "/webhook/wa-tenant/whatsapp",
                json={"entry": []},
                headers={"X-Hub-Signature-256": "sha256=badsig"},
            )
            assert r.status_code == 403
        finally:
            main_module.app.dependency_overrides.pop(get_db, None)


# ─── admin routes: update + queries + contact URL ──────────────────────────────

class TestAdminUpdateAndQueries:
    def test_admin_update_tenant_success(self, api_client):
        import main as main_module
        from db import get_db, Tenant

        mock_tenant = MagicMock(spec=Tenant)
        mock_tenant.id = 1
        mock_tenant.slug = "update-me"
        mock_tenant.bot_token = "tok123"
        mock_tenant.expertise_area = ""
        mock_tenant.active = True
        mock_tenant.wa_phone_number_id = None
        mock_tenant.wa_access_token = None
        mock_tenant.wa_app_secret = None
        mock_tenant.wa_verify_token = None
        mock_tenant.wa_reengagement_template = None
        mock_tenant.channels = "telegram"

        db_override, mock_db = _make_db_mock(scalars_all=[mock_tenant], scalar_one_or_none=mock_tenant)
        main_module.app.dependency_overrides[get_db] = db_override
        try:
            with patch.object(config.settings, "admin_password", "changeme"):
                r = api_client.post(
                    "/admin/tenant/1",
                    data={"expertise_area": "fitness"},
                    headers=_admin_auth("changeme"),
                )
            assert r.status_code == 200
            assert b"actualizado" in r.content.lower()
        finally:
            main_module.app.dependency_overrides.pop(get_db, None)

    def test_admin_update_tenant_invalid_contact_url(self, api_client):
        import main as main_module
        from db import get_db, Tenant

        mock_tenant = MagicMock(spec=Tenant)
        mock_tenant.id = 1
        mock_tenant.slug = "bad-url"
        mock_tenant.active = True

        db_override, _ = _make_db_mock(scalars_all=[mock_tenant], scalar_one_or_none=mock_tenant)
        main_module.app.dependency_overrides[get_db] = db_override
        try:
            with patch.object(config.settings, "admin_password", "changeme"):
                r = api_client.post(
                    "/admin/tenant/1",
                    data={"expertise_area": "fitness", "contact_url": "ftp://bad.com"},
                    headers=_admin_auth("changeme"),
                )
            assert r.status_code == 200
            assert b"http://" in r.content or b"https://" in r.content
        finally:
            main_module.app.dependency_overrides.pop(get_db, None)

    def test_admin_queries_page(self, api_client):
        import main as main_module
        from db import get_db, Tenant

        mock_tenant = MagicMock(spec=Tenant)
        mock_tenant.id = 1
        mock_tenant.slug = "queries-tenant"
        mock_tenant.active = True

        mock_query = MagicMock()
        mock_query.question = "test question"
        mock_query.intent_category = "off_topic"
        mock_query.created_at = datetime.now(timezone.utc)

        db_override, mock_db = _make_db_mock(
            scalars_all=[mock_tenant],
            scalar_one_or_none=mock_tenant
        )
        # Override execute to also return queries for the second call
        mock_result_queries = MagicMock()
        mock_result_queries.scalars.return_value.all.return_value = [mock_query]

        call_count = [0]
        original_execute = mock_db.execute

        async def custom_execute(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                # First call: tenant lookup
                mock_r = MagicMock()
                mock_r.scalar_one_or_none.return_value = mock_tenant
                return mock_r
            # Second call: queries
            return mock_result_queries

        mock_db.execute = custom_execute
        main_module.app.dependency_overrides[get_db] = db_override
        try:
            with patch.object(config.settings, "admin_password", "changeme"):
                r = api_client.get(
                    "/admin/queries/1",
                    headers=_admin_auth("changeme"),
                )
            assert r.status_code == 200
            assert b"queries-tenant" in r.content
        finally:
            main_module.app.dependency_overrides.pop(get_db, None)


# ─── services/upload: detect_mime ─────────────────────────────────────────────

class TestDetectMime:
    def test_detect_mime_fallback_default(self):
        from services.upload import detect_mime
        # filetype may not be installed; verify it returns a string fallback
        mime = detect_mime(b"random bytes that are not an image at all")
        assert isinstance(mime, str)
        assert mime.startswith("image/")  # defaults to image/jpeg

    def test_detect_mime_jpeg_fallback(self):
        from services.upload import detect_mime
        # Without filetype, falls back to "image/jpeg"
        mime = detect_mime(b"\xff\xd8\xff\xe0" + b"\x00" * 50, "image/jpeg")
        assert mime == "image/jpeg"


# ─── rag: sync_faq_chunks ─────────────────────────────────────────────────────

class TestSyncFaqChunks:
    @pytest.mark.asyncio
    async def test_sync_faq_chunks_deletes_and_indexes(self):
        from rag import sync_faq_chunks

        mock_db = AsyncMock()
        mock_db.execute = AsyncMock()
        mock_db.commit = AsyncMock()

        with patch("rag.call_embeddings", new=AsyncMock(return_value=[[0.1] * 1536])):
            with patch("rag.index_chunks", new=AsyncMock(return_value=2)) as mock_index:
                count = await sync_faq_chunks(mock_db, "my-tenant", ["Q1?", "Q2?"])

        mock_index.assert_called_once()
        assert mock_db.execute.called  # DELETE was issued

    @pytest.mark.asyncio
    async def test_sync_faq_chunks_empty_questions_returns_zero(self):
        from rag import sync_faq_chunks

        mock_db = AsyncMock()
        mock_db.execute = AsyncMock()
        mock_db.commit = AsyncMock()

        count = await sync_faq_chunks(mock_db, "my-tenant", None)
        assert count == 0

        count2 = await sync_faq_chunks(mock_db, "my-tenant", [])
        assert count2 == 0


# ─── rag: save_turn ───────────────────────────────────────────────────────────

class TestSaveTurn:
    @pytest.mark.asyncio
    async def test_save_turn_adds_two_rows_and_commits(self):
        from rag import save_turn

        mock_db = AsyncMock()
        mock_db.add = MagicMock()
        mock_db.commit = AsyncMock()
        mock_db.execute = AsyncMock()

        await save_turn(mock_db, "user_1", "ns_1", "Hello", "Hi there", channel="whatsapp")

        assert mock_db.add.call_count == 2  # user + assistant
        assert mock_db.commit.call_count == 2  # add + trim


# ─── rag: _log_unanswered direct ──────────────────────────────────────────────

class TestLogUnanswered:
    @pytest.mark.asyncio
    async def test_log_unanswered_creates_record(self):
        from rag import _log_unanswered

        mock_db = AsyncMock()
        mock_db.add = MagicMock()
        mock_db.commit = AsyncMock()

        await _log_unanswered(mock_db, "ns", "What is X?", "user_1", "off_topic", tenant_id=5)

        mock_db.add.assert_called_once()
        mock_db.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_log_unanswered_exception_does_not_raise(self):
        from rag import _log_unanswered

        mock_db = AsyncMock()
        mock_db.add = MagicMock()
        mock_db.commit = AsyncMock(side_effect=Exception("DB error"))

        # Should not raise
        await _log_unanswered(mock_db, "ns", "Q?", "u1", "off_topic")


# ─── services/stt: 401 and generic error ───────────────────────────────────────

class TestSttAdditional:
    @pytest.mark.asyncio
    async def test_transcribe_voice_401_auth_error(self):
        from services.stt import transcribe_voice

        mock_resp = MagicMock()
        mock_resp.status_code = 401
        mock_resp.text = "Unauthorized"
        mock_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "401", request=MagicMock(), response=mock_resp
        )
        mock_http = AsyncMock()
        mock_http.post = AsyncMock(return_value=mock_resp)

        with patch("services.stt.http_client", mock_http), \
             patch("services.stt.settings") as mock_settings:
            mock_settings.groq_api_key = "bad-key"
            with pytest.raises(RuntimeError, match="authentication"):
                await transcribe_voice(b"audio bytes")

    @pytest.mark.asyncio
    async def test_transcribe_voice_generic_error(self):
        from services.stt import transcribe_voice

        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.text = "Internal Server Error"
        mock_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "500", request=MagicMock(), response=mock_resp
        )
        mock_http = AsyncMock()
        mock_http.post = AsyncMock(return_value=mock_resp)

        with patch("services.stt.http_client", mock_http), \
             patch("services.stt.settings") as mock_settings:
            mock_settings.groq_api_key = "key"
            with pytest.raises(RuntimeError, match="500"):
                await transcribe_voice(b"audio bytes")

    @pytest.mark.asyncio
    async def test_transcribe_voice_no_api_key(self):
        from services.stt import transcribe_voice

        with patch("services.stt.settings") as mock_settings:
            mock_settings.groq_api_key = ""
            with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
                await transcribe_voice(b"audio bytes")


# ─── limiter: rate_limit_handler ──────────────────────────────────────────────

class TestRateLimitHandler:
    def test_rate_limit_handler_returns_429(self):
        from limiter import rate_limit_handler
        from slowapi.errors import RateLimitExceeded
        from slowapi.wrappers import Limit
        from slowapi.util import get_remote_address

        # Build a proper Limit object
        limit_obj = Limit(
            "10 per 1 minute",
            key_func=get_remote_address,
            scope=None,
            per_method=False,
            methods=None,
            error_message=None,
            exempt_when=None,
            cost=1,
            override_defaults=False,
        )
        exc = RateLimitExceeded(limit_obj)
        mock_request = MagicMock()
        response = rate_limit_handler(mock_request, exc)
        assert response.status_code == 429
        import json
        body = json.loads(response.body)
        assert body["detail"] == "Rate limit exceeded"


# ─── Helper for admin tests ───────────────────────────────────────────────────

def _make_db_mock(fetchall=None, scalars_all=None, scalar_one_or_none=None):
    """Return (override_fn, mock_db) for use with app.dependency_overrides[get_db]."""
    mock_result = MagicMock()
    mock_result.fetchall.return_value = fetchall if fetchall is not None else []
    mock_result.scalars.return_value.all.return_value = scalars_all if scalars_all is not None else []
    mock_result.scalar_one_or_none.return_value = scalar_one_or_none

    mock_db = AsyncMock()
    mock_db.execute = AsyncMock(return_value=mock_result)
    mock_db.add = MagicMock()
    mock_db.commit = AsyncMock()
    mock_db.refresh = AsyncMock()

    async def _override():
        yield mock_db

    return _override, mock_db


def _admin_auth(password="changeme"):
    creds = base64.b64encode(f"admin:{password}".encode()).decode()
    return {"Authorization": f"Basic {creds}"}


import config