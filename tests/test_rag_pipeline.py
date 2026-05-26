"""
Smoke tests for the RAG pipeline.

Unit tests run always (no external deps).
Integration tests require live services — run with:
    docker compose up -d
    docker compose exec api python -m pytest tests/ -v
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest


# ─── Unit: text chunking ──────────────────────────────────────────────────────

def test_chunk_text_basic():
    from rag import chunk_text
    text = "A" * 600
    chunks = chunk_text(text, source="test.txt", page=1)
    assert len(chunks) >= 1
    assert all(len(c["content"]) <= 500 for c in chunks)
    assert all(c["source"] == "test.txt" for c in chunks)


def test_chunk_text_skips_tiny_chunks():
    from rag import chunk_text
    chunks = chunk_text("hi", source="test.txt", page=1)
    assert chunks == []


def test_chunk_text_overlap():
    from rag import chunk_text
    text = "X" * 1100
    chunks = chunk_text(text, source="test.txt", page=1)
    assert len(chunks) >= 2
    assert chunks[1]["content"][0] == "X"


def test_chunk_text_empty_string():
    from rag import chunk_text
    assert chunk_text("", source="test.txt", page=1) == []


def test_chunk_text_whitespace_only():
    from rag import chunk_text
    assert chunk_text("   \n  \t  ", source="test.txt", page=1) == []


def test_chunk_text_exact_chunk_size():
    from rag import chunk_text
    chunks = chunk_text("A" * 500, source="test.txt", page=1)
    assert len(chunks) == 1
    assert len(chunks[0]["content"]) == 500


# ─── Unit: generate_answer short-circuits on empty context ───────────────────

@pytest.mark.asyncio
async def test_generate_answer_no_context():
    from rag import generate_answer
    result = await generate_answer([], "What hours are you open?", [])
    assert "no encontré" in result.lower()


# ─── Unit: index_chunks with empty list ──────────────────────────────────────

@pytest.mark.asyncio
async def test_index_chunks_empty():
    from rag import index_chunks
    assert await index_chunks(None, [], "test_ns") == 0


# ─── Fixtures ────────────────────────────────────────────────────────────────

def _patch_lifespan_db():
    """Context manager that mocks DB calls in the lifespan (no tenants loaded)."""
    from unittest.mock import AsyncMock, patch, MagicMock
    mock_result = MagicMock()
    mock_result.scalars.return_value.all.return_value = []
    mock_db = AsyncMock()
    mock_db.execute = AsyncMock(return_value=mock_result)
    mock_db.__aenter__ = AsyncMock(return_value=mock_db)
    mock_db.__aexit__ = AsyncMock(return_value=False)
    mock_session_local = MagicMock(return_value=mock_db)
    return patch("db.AsyncSessionLocal", mock_session_local)


@pytest.fixture
def api_client():
    """Unauthenticated client — only for /health and auth-rejection tests."""
    from unittest.mock import AsyncMock, patch
    import main as main_module
    from fastapi.testclient import TestClient
    with patch("main.init_db", new_callable=AsyncMock), _patch_lifespan_db():
        with TestClient(main_module.app) as client:
            yield client


@pytest.fixture
def authed_api_client():
    """Authenticated client — mocks require_tenant to return a test Tenant."""
    from unittest.mock import AsyncMock, patch, MagicMock
    import main as main_module
    from fastapi.testclient import TestClient
    from db import Tenant

    mock_tenant = MagicMock(spec=Tenant)
    mock_tenant.slug = "test-tenant"
    mock_tenant.active = True

    async def _mock_require_tenant(*args, **kwargs):
        return mock_tenant

    with patch("main.init_db", new_callable=AsyncMock), _patch_lifespan_db():
        with patch("main.require_tenant", side_effect=_mock_require_tenant):
            with TestClient(main_module.app) as client:
                yield client, mock_tenant


# ─── API: /health ─────────────────────────────────────────────────────────────

def test_health_endpoint(api_client):
    r = api_client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


# ─── API: auth required ───────────────────────────────────────────────────────

def test_upload_requires_auth(api_client):
    r = api_client.post(
        "/upload",
        files={"file": ("test.pdf", b"content", "application/pdf")},
    )
    assert r.status_code == 422  # missing X-API-Key header → FastAPI validation error


def test_stats_requires_auth(api_client):
    r = api_client.get("/stats")
    assert r.status_code == 422


def test_delete_namespace_requires_auth(api_client):
    r = api_client.delete("/namespace")
    assert r.status_code == 422


# ─── API: upload validation (with auth) ──────────────────────────────────────

def test_upload_rejects_non_pdf(authed_api_client):
    client, _ = authed_api_client
    r = client.post(
        "/upload",
        files={"file": ("readme.txt", b"some text", "text/plain")},
        headers={"X-API-Key": "test-key"},
    )
    assert r.status_code == 400
    assert "PDF" in r.json()["detail"]


def test_upload_accepts_uppercase_extension(authed_api_client):
    client, _ = authed_api_client
    r = client.post(
        "/upload",
        files={"file": ("DOCUMENT.PDF", b"fake content", "application/pdf")},
        headers={"X-API-Key": "test-key"},
    )
    assert r.json().get("detail") != "Only PDF files are supported"


def test_upload_rejects_oversized_file(authed_api_client):
    client, _ = authed_api_client
    oversized = b"0" * (11 * 1024 * 1024)
    r = client.post(
        "/upload",
        files={"file": ("big.pdf", oversized, "application/pdf")},
        headers={"X-API-Key": "test-key"},
    )
    assert r.status_code == 413


def test_upload_rejects_corrupted_pdf(authed_api_client):
    client, _ = authed_api_client
    r = client.post(
        "/upload",
        files={"file": ("bad.pdf", b"not-a-real-pdf", "application/pdf")},
        headers={"X-API-Key": "test-key"},
    )
    assert r.status_code == 400
    assert "Could not read PDF" in r.json()["detail"]


# ─── API: webhook security ────────────────────────────────────────────────────

def test_webhook_returns_404_unknown_tenant(api_client):
    from unittest.mock import AsyncMock, MagicMock
    import main as main_module
    from db import get_db

    mock_db = AsyncMock()
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = None
    mock_db.execute = AsyncMock(return_value=mock_result)

    async def _mock_get_db():
        yield mock_db

    main_module.app.dependency_overrides[get_db] = _mock_get_db
    try:
        r = api_client.post(
            "/webhook/nonexistent-tenant",
            json={"update_id": 1},
            headers={"X-Telegram-Bot-Api-Secret-Token": "any"},
        )
        assert r.status_code == 404
    finally:
        main_module.app.dependency_overrides.pop(get_db, None)


def test_webhook_rejects_invalid_signature(api_client):
    from unittest.mock import AsyncMock, MagicMock
    import main as main_module
    from db import get_db, Tenant

    mock_tenant = MagicMock(spec=Tenant)
    mock_tenant.slug = "test-tenant"
    mock_tenant.webhook_secret = "correct-secret"
    mock_tenant.bot_token = "fake-token"
    mock_tenant.active = True

    mock_db = AsyncMock()
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = mock_tenant
    mock_db.execute = AsyncMock(return_value=mock_result)

    async def _mock_get_db():
        yield mock_db

    main_module.app.dependency_overrides[get_db] = _mock_get_db
    try:
        r = api_client.post(
            "/webhook/test-tenant",
            json={"update_id": 1},
            headers={"X-Telegram-Bot-Api-Secret-Token": "wrong-secret"},
        )
        assert r.status_code == 403
    finally:
        main_module.app.dependency_overrides.pop(get_db, None)


# ─── Integration: full pipeline ───────────────────────────────────────────────
# Requires: docker compose up -d && postgres + api healthy
# Run: docker compose exec api python -m pytest tests/ -v -m integration

@pytest.mark.integration
@pytest.mark.asyncio
async def test_embed_and_retrieve(tmp_path):
    import asyncio
    from db import init_db, AsyncSessionLocal
    from rag import index_chunks, retrieve_context

    await init_db()

    chunks = [{"content": "The gym is open on Sundays from 8am to 8pm.", "source": "smoke_test.txt", "page": 1}]

    async with AsyncSessionLocal() as db:
        stored = await index_chunks(db, chunks, namespace="_smoke_test")
        assert stored == 1

        results = await retrieve_context(db, "What are Sunday hours?", namespace="_smoke_test", top_k=1)
        assert len(results) == 1
        assert results[0]["similarity"] > 0.5

        from sqlalchemy import text
        await db.execute(text("DELETE FROM document_chunks WHERE namespace = '_smoke_test'"))
        await db.commit()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_cross_tenant_isolation():
    """Chunks indexed under tenant_a are not visible from tenant_b namespace."""
    from db import AsyncSessionLocal
    from rag import index_chunks, retrieve_context

    chunks = [{"content": "Secret pricing: $999/month for enterprise.", "source": "secret.txt", "page": 1}]

    async with AsyncSessionLocal() as db:
        await index_chunks(db, chunks, namespace="_tenant_a_isolation_test")
        results = await retrieve_context(db, "pricing", namespace="_tenant_b_isolation_test", top_k=5)
        assert all(r["similarity"] < 0.9 for r in results), "Cross-tenant data leak detected"

        from sqlalchemy import text
        await db.execute(text("DELETE FROM document_chunks WHERE namespace LIKE '_tenant_%_isolation_test'"))
        await db.commit()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_delete_only_deletes_own_namespace():
    """DELETE /namespace only removes the authenticated tenant's data."""
    from db import AsyncSessionLocal, DocumentChunk
    from rag import index_chunks
    from sqlalchemy import select, func

    chunks_a = [{"content": "Tenant A document.", "source": "a.txt", "page": 1}]
    chunks_b = [{"content": "Tenant B document.", "source": "b.txt", "page": 1}]

    async with AsyncSessionLocal() as db:
        await index_chunks(db, chunks_a, namespace="_delete_test_a")
        await index_chunks(db, chunks_b, namespace="_delete_test_b")

        from sqlalchemy import text
        await db.execute(text("DELETE FROM document_chunks WHERE namespace = '_delete_test_a'"))
        await db.commit()

        result = await db.execute(
            select(func.count(DocumentChunk.id)).where(DocumentChunk.namespace == "_delete_test_b")
        )
        assert result.scalar() == 1, "Tenant B data was incorrectly deleted"

        await db.execute(text("DELETE FROM document_chunks WHERE namespace = '_delete_test_b'"))
        await db.commit()
