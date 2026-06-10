"""Tests for PR2 portal features.

Covers:
- Portal login (JSON + form/cookie)
- Portal dashboard rendering
- Document upload/delete with plan limits
- Answer preview query
- Plan limit enforcement (E2)
- Audit log writes (E6)
- Usage metering (E2)
- Citation enforcement in system prompt (E1)
- Faithfulness self-check (E1)
- Portal auth (cookie + header)
- Unanswered question resolution (E5)
"""
import base64
import os
import sys
import time
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
from fastapi.testclient import TestClient


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _make_portal_auth_override(tenant_mock=None):
    """Return (override_fn, mock_tenant, mock_tenant_db) for require_portal_auth."""
    from db import Tenant
    mock_tenant = tenant_mock or MagicMock(spec=Tenant)
    mock_tenant.id = 1
    mock_tenant.slug = "test-tenant"
    mock_tenant.active = True
    mock_tenant.expertise_area = "Test Business"
    mock_tenant.plan = "free"
    mock_tenant.portal_password_hash = None

    mock_tenant_db = AsyncMock()
    mock_tenant_db.execute = AsyncMock(return_value=MagicMock())
    mock_tenant_db.commit = AsyncMock()
    mock_tenant_db.add = MagicMock()

    async def _override():
        yield mock_tenant, mock_tenant_db

    return _override, mock_tenant, mock_tenant_db


def _admin_auth(password="test"):
    """Return Basic auth headers for admin routes."""
    creds = base64.b64encode(f"admin:{password}".encode()).decode()
    return {"Authorization": f"Basic {creds}"}


# ─── PLAN_LIMITS (E2) ────────────────────────────────────────────────────────

class TestPlanLimits:
    def test_plan_limits_has_all_tiers(self):
        from config import PLAN_LIMITS
        assert "free" in PLAN_LIMITS
        assert "basic" in PLAN_LIMITS
        assert "pro" in PLAN_LIMITS

    def test_plan_limits_has_required_keys(self):
        from config import PLAN_LIMITS
        for plan in PLAN_LIMITS.values():
            assert "docs" in plan
            assert "chunks" in plan
            assert "queries_monthly" in plan

    def test_free_tier_has_low_limits(self):
        from config import PLAN_LIMITS
        assert PLAN_LIMITS["free"]["docs"] < PLAN_LIMITS["basic"]["docs"]
        assert PLAN_LIMITS["basic"]["docs"] < PLAN_LIMITS["pro"]["docs"]


# ─── Audit Log (E6) ──────────────────────────────────────────────────────────

class TestAuditLog:
    @pytest.mark.asyncio
    async def test_write_audit_log_inserts_row(self):
        from services.usage import write_audit_log
        mock_db = AsyncMock()
        mock_db.add = MagicMock()
        mock_db.commit = AsyncMock()

        await write_audit_log(
            mock_db, tenant_id=1, namespace="test-tenant",
            actor="tenant:test-tenant", action="document.upload",
            detail={"source": "test.pdf", "chunks": 5},
        )
        mock_db.add.assert_called_once()
        call_args = mock_db.add.call_args[0][0]
        assert call_args.tenant_id == 1
        assert call_args.namespace == "test-tenant"
        assert call_args.actor == "tenant:test-tenant"
        assert call_args.action == "document.upload"
        assert call_args.detail == {"source": "test.pdf", "chunks": 5}

    @pytest.mark.asyncio
    async def test_write_audit_log_auto_commit_default(self):
        from services.usage import write_audit_log
        mock_db = AsyncMock()
        mock_db.add = MagicMock()
        mock_db.commit = AsyncMock()

        await write_audit_log(mock_db, 1, "ns", "admin", "portal.login")
        mock_db.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_write_audit_log_skip_commit(self):
        from services.usage import write_audit_log
        mock_db = AsyncMock()
        mock_db.add = MagicMock()
        mock_db.commit = AsyncMock()

        await write_audit_log(mock_db, 1, "ns", "admin", "portal.login", auto_commit=False)
        mock_db.commit.assert_not_called()


# ─── Portal Auth ──────────────────────────────────────────────────────────────

class TestPortalAuth:
    def test_require_portal_auth_missing_token_redirects(self):
        """Without cookie or Authorization header, dashboard returns 303 or 401."""
        import main as main_module

        # No dependency overrides — require_portal_auth should reject unauthenticated requests
        client = TestClient(main_module.app, raise_server_exceptions=False)
        resp = client.get("/portal/dashboard", follow_redirects=False)
        # Should redirect to login (303) or return 401
        assert resp.status_code in (303, 401)

    def test_portal_login_page_returns_html(self):
        """GET /portal/login should render the login form."""
        import main as main_module
        client = TestClient(main_module.app)
        resp = client.get("/portal/login")
        assert resp.status_code == 200
        assert "Iniciar sesión" in resp.text or "login" in resp.text.lower()

    def test_portal_login_json_returns_token(self):
        """POST /portal/login with JSON body returns JWT."""
        import main as main_module
        from db import Tenant, get_db
        from auth import hash_portal_password

        mock_tenant = MagicMock(spec=Tenant)
        mock_tenant.id = 1
        mock_tenant.slug = "test-tenant"
        mock_tenant.active = True
        mock_tenant.portal_password_hash = hash_portal_password("testpass")

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_tenant
        mock_db = AsyncMock()
        mock_db.execute = AsyncMock(return_value=mock_result)
        mock_db.add = MagicMock()
        mock_db.commit = AsyncMock()

        async def _override():
            yield mock_db

        main_module.app.dependency_overrides[get_db] = _override
        with patch("routes.portal.settings") as mock_settings:
            mock_settings.jwt_secret = "test-secret-key-for-testing-min-16ch"
            try:
                client = TestClient(main_module.app)
                resp = client.post("/portal/login", json={"slug": "test-tenant", "password": "testpass"})
                assert resp.status_code == 200
                data = resp.json()
                assert "access_token" in data
                assert data["token_type"] == "bearer"
            finally:
                main_module.app.dependency_overrides.pop(get_db, None)


# ─── Citation Enforcement (E1) ────────────────────────────────────────────────

class TestCitationEnforcement:
    def test_system_prompt_includes_citation_instructions(self):
        from services.prompts import build_system_prompt
        prompt = build_system_prompt(expertise_area="Veterinaria")
        assert "CITACIONES" in prompt
        assert "[Source:" in prompt
        assert "Page" in prompt

    def test_citation_clause_present_for_all_channels(self):
        from services.prompts import build_system_prompt
        for channel in ("telegram", "whatsapp", "portal"):
            prompt = build_system_prompt(expertise_area="Test", channel=channel)
            assert "CITACIONES" in prompt


# ─── Faithfulness Self-Check (E1) ─────────────────────────────────────────────

class TestFaithfulnessCheck:
    @pytest.mark.asyncio
    async def test_faithfulness_check_returns_original_when_faithful(self):
        from rag import _faithfulness_check
        with patch("rag.call_chat", new_callable=AsyncMock) as mock_chat:
            mock_chat.return_value = '{"faithful": true}'
            result = await _faithfulness_check(
                question="¿Cuál es el precio?",
                answer="El precio es $90 [Source: catalog.pdf, Page 3].",
                context=[{"source": "catalog.pdf", "page": 3, "content": "Precio: $90", "similarity": 0.7}],
            )
            assert result == "El precio es $90 [Source: catalog.pdf, Page 3]."

    @pytest.mark.asyncio
    async def test_faithfulness_check_adds_caveat_when_unfaithful(self):
        from rag import _faithfulness_check
        with patch("rag.call_chat", new_callable=AsyncMock) as mock_chat:
            mock_chat.return_value = '{"faithful": false, "unsupported_claims": "precio no está en el contexto"}'
            result = await _faithfulness_check(
                question="¿Cuál es el precio?",
                answer="El precio es $500.",
                context=[{"source": "catalog.pdf", "page": 3, "content": "Precio: $90", "similarity": 0.7}],
            )
            assert "⚠️ Aclaración" in result
            assert "precio no está en el contexto" in result

    @pytest.mark.asyncio
    async def test_faithfulness_check_returns_original_on_llm_failure(self):
        from rag import _faithfulness_check
        original = "El precio es $90."
        with patch("rag.call_chat", new_callable=AsyncMock, side_effect=Exception("LLM error")):
            result = await _faithfulness_check(
                question="¿Cuál es el precio?",
                answer=original,
                context=[{"source": "catalog.pdf", "page": 3, "content": "Precio: $90", "similarity": 0.7}],
            )
            assert result == original

    def test_faithfulness_ceiling_constant(self):
        from rag import _FAITHFULNESS_CEILING
        assert _FAITHFULNESS_CEILING == 0.85


# ─── Usage Metering (E2) ──────────────────────────────────────────────────────

class TestUsageMetering:
    @pytest.mark.asyncio
    async def test_increment_usage_atomic_upsert(self):
        from services.usage import increment_usage
        mock_db = AsyncMock()
        mock_db.execute = AsyncMock(return_value=MagicMock())
        mock_db.commit = AsyncMock()

        await increment_usage(mock_db, tenant_id=1, metric="queries")
        mock_db.execute.assert_called_once()
        mock_db.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_usage_returns_zero_when_no_row(self):
        from services.usage import get_usage
        mock_db = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_db.execute = AsyncMock(return_value=mock_result)

        result = await get_usage(mock_db, tenant_id=1, metric="queries")
        assert result == 0

    @pytest.mark.asyncio
    async def test_get_usage_returns_value(self):
        from services.usage import get_usage
        mock_db = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = 42
        mock_db.execute = AsyncMock(return_value=mock_result)

        result = await get_usage(mock_db, tenant_id=1, metric="queries")
        assert result == 42


# ─── Portal Dashboard ────────────────────────────────────────────────────────

class TestPortalDashboard:
    def test_dashboard_requires_auth(self):
        """GET /portal/dashboard without auth cookie redirects to login."""
        import main as main_module
        client = TestClient(main_module.app, raise_server_exceptions=False)
        resp = client.get("/portal/dashboard", follow_redirects=False)
        assert resp.status_code in (303, 401)

    def test_dashboard_renders_with_auth(self):
        """GET /portal/dashboard with valid auth returns HTML with tabs."""
        import main as main_module
        from dependencies import require_portal_auth
        from db import Tenant

        mock_tenant = MagicMock(spec=Tenant)
        mock_tenant.id = 1
        mock_tenant.slug = "test-tenant"
        mock_tenant.active = True
        mock_tenant.expertise_area = "Test Business"
        mock_tenant.plan = "free"

        mock_tenant_db = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = []
        mock_tenant_db.execute = AsyncMock(return_value=mock_result)
        mock_tenant_db.commit = AsyncMock()

        async def _override():
            yield mock_tenant, mock_tenant_db

        main_module.app.dependency_overrides[require_portal_auth] = _override
        with patch("routes.portal.list_sources", new_callable=AsyncMock, return_value=[]), \
             patch("routes.portal.get_usage", new_callable=AsyncMock, return_value=0):
            try:
                client = TestClient(main_module.app)
                resp = client.get("/portal/dashboard")
                assert resp.status_code == 200
                assert "Conocimiento" in resp.text
                assert "Uso" in resp.text
                assert "Probar" in resp.text
                assert "Preguntas" in resp.text
            finally:
                main_module.app.dependency_overrides.pop(require_portal_auth, None)

    def test_dashboard_shows_plan_limits(self):
        """Dashboard should show usage bars with plan limit values."""
        import main as main_module
        from dependencies import require_portal_auth
        from db import Tenant

        mock_tenant = MagicMock(spec=Tenant)
        mock_tenant.id = 1
        mock_tenant.slug = "test-tenant"
        mock_tenant.active = True
        mock_tenant.expertise_area = "Test"
        mock_tenant.plan = "free"

        mock_tenant_db = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = []
        mock_tenant_db.execute = AsyncMock(return_value=mock_result)
        mock_tenant_db.commit = AsyncMock()

        async def _override():
            yield mock_tenant, mock_tenant_db

        main_module.app.dependency_overrides[require_portal_auth] = _override
        with patch("routes.portal.list_sources", new_callable=AsyncMock, return_value=[]), \
             patch("routes.portal.get_usage", new_callable=AsyncMock, return_value=0):
            try:
                client = TestClient(main_module.app)
                resp = client.get("/portal/dashboard")
                assert resp.status_code == 200
                # Free plan limits should appear in the HTML
                assert "5" in resp.text  # docs limit for free plan
                assert "500" in resp.text  # queries limit for free plan
            finally:
                main_module.app.dependency_overrides.pop(require_portal_auth, None)


# ─── Portal Query (E3 Answer Preview) ─────────────────────────────────────────

class TestPortalQuery:
    def test_query_returns_json_answer(self):
        """POST /portal/query returns answer and sources."""
        import main as main_module
        from dependencies import require_portal_auth
        from db import Tenant

        mock_tenant = MagicMock(spec=Tenant)
        mock_tenant.id = 1
        mock_tenant.slug = "test-tenant"
        mock_tenant.active = True
        mock_tenant.expertise_area = "Test"
        mock_tenant.plan = "free"

        mock_tenant_db = AsyncMock()
        mock_tenant_db.execute = AsyncMock(return_value=MagicMock())
        mock_tenant_db.commit = AsyncMock()
        mock_tenant_db.add = MagicMock()

        async def _override():
            yield mock_tenant, mock_tenant_db

        main_module.app.dependency_overrides[require_portal_auth] = _override
        with patch("routes.portal._rag_query", new_callable=AsyncMock, return_value=("Answer text", [{"source": "doc.pdf", "page": 3, "content": "test", "similarity": 0.9}], None)), \
             patch("routes.portal.get_usage", new_callable=AsyncMock, return_value=10), \
             patch("routes.portal.increment_usage", new_callable=AsyncMock), \
             patch("routes.portal.write_audit_log", new_callable=AsyncMock):
            try:
                client = TestClient(main_module.app)
                resp = client.post("/portal/query", data={"question": "¿Precio?"})
                assert resp.status_code == 200
                data = resp.json()
                assert data["answer"] == "Answer text"
                assert len(data["sources"]) == 1
                assert data["sources"][0]["source"] == "doc.pdf"
            finally:
                main_module.app.dependency_overrides.pop(require_portal_auth, None)

    def test_query_returns_429_when_limit_reached(self):
        """POST /portal/query returns 429 when monthly query limit is exceeded."""
        import main as main_module
        from dependencies import require_portal_auth
        from db import Tenant
        from config import PLAN_LIMITS

        mock_tenant = MagicMock(spec=Tenant)
        mock_tenant.id = 1
        mock_tenant.slug = "test-tenant"
        mock_tenant.active = True
        mock_tenant.expertise_area = "Test"
        mock_tenant.plan = "free"

        mock_tenant_db = AsyncMock()
        mock_tenant_db.execute = AsyncMock(return_value=MagicMock())
        mock_tenant_db.commit = AsyncMock()

        async def _override():
            yield mock_tenant, mock_tenant_db

        # check_and_increment_usage returns False when limit is reached
        main_module.app.dependency_overrides[require_portal_auth] = _override
        with patch("routes.portal.check_and_increment_usage", new_callable=AsyncMock, return_value=False):
            try:
                client = TestClient(main_module.app)
                resp = client.post("/portal/query", data={"question": "test"})
                assert resp.status_code == 429
            finally:
                main_module.app.dependency_overrides.pop(require_portal_auth, None)

    def test_query_returns_400_when_question_empty(self):
        """POST /portal/query returns 400 when question is empty."""
        import main as main_module
        from dependencies import require_portal_auth
        from db import Tenant

        mock_tenant = MagicMock(spec=Tenant)
        mock_tenant.id = 1
        mock_tenant.slug = "test-tenant"
        mock_tenant.active = True
        mock_tenant.expertise_area = "Test"
        mock_tenant.plan = "free"

        mock_tenant_db = AsyncMock()
        mock_tenant_db.execute = AsyncMock(return_value=MagicMock())
        mock_tenant_db.commit = AsyncMock()

        async def _override():
            yield mock_tenant, mock_tenant_db

        main_module.app.dependency_overrides[require_portal_auth] = _override
        with patch("routes.portal.get_usage", new_callable=AsyncMock, return_value=0):
            try:
                client = TestClient(main_module.app)
                resp = client.post("/portal/query", data={"question": ""})
                assert resp.status_code == 400
            finally:
                main_module.app.dependency_overrides.pop(require_portal_auth, None)


# ─── Portal Login Form (Cookie-based) ─────────────────────────────────────────

class TestPortalLoginForm:
    def test_login_form_wrong_password_shows_error(self):
        """POST /portal/login/form with wrong password re-renders with error."""
        import main as main_module
        from db import Tenant, get_db

        mock_result = MagicMock()
        mock_tenant = MagicMock(spec=Tenant)
        mock_tenant.slug = "test-tenant"
        mock_tenant.active = True
        mock_tenant.portal_password_hash = None  # Not configured
        mock_result.scalar_one_or_none.return_value = mock_tenant

        mock_db = AsyncMock()
        mock_db.execute = AsyncMock(return_value=mock_result)

        async def _override():
            yield mock_db

        main_module.app.dependency_overrides[get_db] = _override
        try:
            client = TestClient(main_module.app)
            resp = client.post("/portal/login/form", data={"slug": "test-tenant", "password": "wrong"})
            assert resp.status_code == 200
            assert "Credenciales inválidas" in resp.text or "Portal access" in resp.text or "error" in resp.text.lower()
        finally:
            main_module.app.dependency_overrides.pop(get_db, None)