"""
Shared test fixtures — session-scoped TestClient avoids 20-60s ngrok polling per test.

Root cause of slow tests: lifespan calls _get_ngrok_domain (20 retries × 3s = 60s)
on every TestClient creation. Patching it + session-scoping the client eliminates this.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient


def _patch_lifespan_db():
    """Mock DB calls in the lifespan so no tenants are loaded."""
    mock_result = MagicMock()
    mock_result.scalars.return_value.all.return_value = []
    mock_db = AsyncMock()
    mock_db.execute = AsyncMock(return_value=mock_result)
    mock_db.__aenter__ = AsyncMock(return_value=mock_db)
    mock_db.__aexit__ = AsyncMock(return_value=False)
    return patch("db.AsyncSessionLocal", MagicMock(return_value=mock_db))


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


@pytest.fixture(scope="session")
def _app_client():
    """Session-scoped TestClient — patches init_db, DB, ngrok, and config overlay to avoid network delays."""
    import main as main_module
    from db import get_db

    # Base mock DB for all requests via dependency override
    _db_override, _mock_db = _make_db_mock()

    with patch("lifespan.init_db", new_callable=AsyncMock), \
         _patch_lifespan_db(), \
         patch("services.ngrok.get_ngrok_domain", new_callable=AsyncMock, return_value="localhost"), \
         patch("config_overlay.reload_from_db", new_callable=AsyncMock):
        main_module.app.dependency_overrides[get_db] = _db_override
        with TestClient(main_module.app) as client:
            yield client


@pytest.fixture
def api_client(_app_client):
    """Unauthenticated API client — reuse session-scoped TestClient."""
    return _app_client


@pytest.fixture
def authed_api_client(_app_client):
    """Authenticated API client — patches require_tenant for the test's duration."""
    import main as main_module
    from db import Tenant, get_db
    from dependencies import require_tenant

    mock_tenant = MagicMock(spec=Tenant)
    mock_tenant.slug = "test-tenant"
    mock_tenant.active = True
    mock_tenant.bot_token = "fake-token"
    mock_tenant.expertise_area = ""

    async def _mock_require_tenant():
        return mock_tenant

    main_module.app.dependency_overrides[require_tenant] = _mock_require_tenant
    yield _app_client, mock_tenant
    main_module.app.dependency_overrides.pop(require_tenant, None)