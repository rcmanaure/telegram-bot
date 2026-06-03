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
from unittest.mock import AsyncMock, patch, MagicMock
import httpx


# ─── Unit: text chunking ──────────────────────────────────────────────────────

def test_chunk_text_basic():
    from rag import chunk_text
    from config import settings
    text = "A" * 600
    chunks = chunk_text(text, source="test.txt", page=1)
    assert len(chunks) >= 1
    assert all(len(c["content"]) <= settings.chunk_size for c in chunks)
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


# api_client and authed_api_client fixtures are defined in conftest.py (session-scoped)


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

def test_upload_rejects_unsupported_format(authed_api_client):
    client, _ = authed_api_client
    r = client.post(
        "/upload",
        files={"file": ("data.xlsx", b"fake", "application/vnd.ms-excel")},
        headers={"X-API-Key": "test-key"},
    )
    assert r.status_code == 400
    assert "Supported formats" in r.json()["detail"]


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


# ─── Unit: call_chat (via llm module) ───────────────────────────────────────────

def _mock_response(status_code=200, json_data=None):
    """Build a fake httpx.Response."""
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status_code
    resp.headers = MagicMock()
    resp.headers.get = MagicMock(return_value="application/json")
    if json_data is not None:
        resp.json = MagicMock(return_value=json_data)
    resp.raise_for_status = MagicMock()
    if status_code >= 400:
        resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "error", request=MagicMock(), response=resp
        )
    return resp


@pytest.mark.asyncio
async def test_call_chat_primary_succeeds():
    from llm import call_chat
    ok_resp = _mock_response(json_data={"choices": [{"message": {"content": "hello"}}]})
    with patch("llm._chat_client") as mock_client, \
         patch("llm.settings") as mock_settings:
        mock_settings.llm_model = "primary-model"
        mock_settings.llm_fallback_model = "fallback-model"
        mock_settings.effective_llm_api_key = "test-key"
        mock_settings.llm_base_url = "https://openrouter.ai/api/v1"
        mock_client.post = AsyncMock(return_value=ok_resp)
        result = await call_chat([{"role": "user", "content": "hi"}])
    assert result == "hello"


@pytest.mark.asyncio
async def test_call_chat_fallback_on_429():
    from llm import call_chat
    fail_resp = _mock_response(status_code=429)
    ok_resp = _mock_response(json_data={"choices": [{"message": {"content": "fallback reply"}}]})
    with patch("llm._chat_client") as mock_client, \
         patch("llm.settings") as mock_settings:
        mock_settings.llm_model = "primary-model"
        mock_settings.llm_fallback_model = "fallback-model"
        mock_settings.effective_llm_api_key = "test-key"
        mock_settings.llm_base_url = "https://openrouter.ai/api/v1"
        mock_client.post = AsyncMock(side_effect=[fail_resp, ok_resp])
        result = await call_chat([{"role": "user", "content": "hi"}])
    assert result == "fallback reply"


@pytest.mark.asyncio
async def test_call_chat_fallback_on_timeout():
    from llm import call_chat
    ok_resp = _mock_response(json_data={"choices": [{"message": {"content": "fallback reply"}}]})
    with patch("llm._chat_client") as mock_client, \
         patch("llm.settings") as mock_settings:
        mock_settings.llm_model = "primary-model"
        mock_settings.llm_fallback_model = "fallback-model"
        mock_settings.effective_llm_api_key = "test-key"
        mock_settings.llm_base_url = "https://openrouter.ai/api/v1"
        mock_client.post = AsyncMock(side_effect=[
            httpx.TimeoutException("timeout"),
            ok_resp,
        ])
        result = await call_chat([{"role": "user", "content": "hi"}])
    assert result == "fallback reply"


@pytest.mark.asyncio
async def test_call_chat_both_fail():
    from llm import call_chat
    fail_429 = _mock_response(status_code=429)
    fail_500 = _mock_response(status_code=500)
    with patch("llm._chat_client") as mock_client, \
         patch("llm.settings") as mock_settings:
        mock_settings.llm_model = "primary-model"
        mock_settings.llm_fallback_model = "fallback-model"
        mock_settings.effective_llm_api_key = "test-key"
        mock_settings.llm_base_url = "https://openrouter.ai/api/v1"
        mock_client.post = AsyncMock(side_effect=[fail_429, fail_500])
        with pytest.raises(RuntimeError, match="LLM service error"):
            await call_chat([{"role": "user", "content": "hi"}])


@pytest.mark.asyncio
async def test_call_chat_fallback_disabled():
    from llm import call_chat
    fail_429 = _mock_response(status_code=429)
    with patch("llm._chat_client") as mock_client, \
         patch("llm.settings") as mock_settings:
        mock_settings.llm_model = "primary-model"
        mock_settings.llm_fallback_model = ""
        mock_settings.effective_llm_api_key = "test-key"
        mock_settings.llm_base_url = "https://openrouter.ai/api/v1"
        mock_client.post = AsyncMock(return_value=fail_429)
        with pytest.raises(RuntimeError, match="rate-limited"):
            await call_chat([{"role": "user", "content": "hi"}])
        # Only one call — no fallback attempt
        assert mock_client.post.call_count == 1


@pytest.mark.asyncio
async def test_call_chat_malformed_response_fallback():
    from llm import call_chat
    bad_resp = _mock_response(json_data={"no_choices": True})
    ok_resp = _mock_response(json_data={"choices": [{"message": {"content": "recovered"}}]})
    with patch("llm._chat_client") as mock_client, \
         patch("llm.settings") as mock_settings:
        mock_settings.llm_model = "primary-model"
        mock_settings.llm_fallback_model = "fallback-model"
        mock_settings.effective_llm_api_key = "test-key"
        mock_settings.llm_base_url = "https://openrouter.ai/api/v1"
        mock_client.post = AsyncMock(side_effect=[bad_resp, ok_resp])
        result = await call_chat([{"role": "user", "content": "hi"}])
    assert result == "recovered"


# ─── Unit: call_embeddings passes dimensions to API (T9-pre) ─────────────────

@pytest.mark.asyncio
async def test_call_embeddings_passes_dimensions_to_api():
    """call_embeddings must always pass dimensions= so MRL truncation and
    validate_config() dimension checks are consistent with EMBEDDING_DIM."""
    from llm import call_embeddings
    mock_response = MagicMock()
    mock_response.data = [MagicMock(embedding=[0.1] * 1536)]

    mock_client = AsyncMock()
    mock_client.embeddings.create = AsyncMock(return_value=mock_response)

    with patch("llm._get_embedding_client", return_value=mock_client), \
         patch("llm.settings") as mock_settings, \
         patch("llm.get_setting_int", return_value=1536):
        mock_settings.embedding_model = "test-model"
        mock_settings.embedding_dim = 1536
        await call_embeddings(["hello"])

    call_kwargs = mock_client.embeddings.create.call_args
    assert "dimensions" in call_kwargs.kwargs, "dimensions kwarg must be passed to embeddings API"
    assert call_kwargs.kwargs["dimensions"] == 1536


# ─── Unit: _hyde_query (T9) ───────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_hyde_query_happy_path():
    from rag import _hyde_query
    with patch("rag.call_chat", new_callable=AsyncMock, return_value="Biopsia de apéndice cecal $90.00"):
        result = await _hyde_query("¿cuánto cuesta una biopsia?", "estudios médicos")
    assert result == "Biopsia de apéndice cecal $90.00"


@pytest.mark.asyncio
async def test_hyde_query_llm_failure_returns_empty():
    from rag import _hyde_query
    with patch("rag.call_chat", new_callable=AsyncMock, side_effect=RuntimeError("timeout")):
        result = await _hyde_query("¿precio de biopsia?", "estudios médicos")
    assert result == ""


@pytest.mark.asyncio
async def test_hyde_query_empty_response_returns_empty():
    from rag import _hyde_query
    with patch("rag.call_chat", new_callable=AsyncMock, return_value="   "):
        result = await _hyde_query("¿precio?", "médico")
    assert result == ""


@pytest.mark.asyncio
async def test_hyde_query_over_500_chars_returns_empty():
    from rag import _hyde_query
    long_text = "A" * 501
    with patch("rag.call_chat", new_callable=AsyncMock, return_value=long_text):
        result = await _hyde_query("pregunta", "médico")
    assert result == ""


@pytest.mark.asyncio
async def test_hyde_query_8_chars_accepted():
    """Short but valid hypotheticals like 'Biopsia.' (8 chars) must be accepted."""
    from rag import _hyde_query
    with patch("rag.call_chat", new_callable=AsyncMock, return_value="Biopsia."):
        result = await _hyde_query("¿qué es una biopsia?", "médico")
    assert result == "Biopsia."


@pytest.mark.asyncio
async def test_hyde_query_2_chars_rejected():
    """Responses shorter than 3 chars are noise — reject them."""
    from rag import _hyde_query
    with patch("rag.call_chat", new_callable=AsyncMock, return_value="Sí"):
        result = await _hyde_query("¿hay biopsias?", "médico")
    assert result == ""


@pytest.mark.asyncio
async def test_hyde_disabled_skips_hyde():
    """hyde_enabled=off must prevent _hyde_query from being called."""
    from rag import rag_query
    mock_tenant = MagicMock()
    mock_tenant.web_search_enabled = False
    mock_tenant.language_code = None

    with patch("rag.get_history", new_callable=AsyncMock, return_value=[]), \
         patch("rag._reformulate_query", new_callable=AsyncMock, return_value="pregunta"), \
         patch("rag._hyde_query", new_callable=AsyncMock, return_value="hipotética") as mock_hyde, \
         patch("rag.retrieve_context", new_callable=AsyncMock, return_value=[]), \
         patch("rag._triage_response", new_callable=AsyncMock, return_value=("off_topic", "lo siento")), \
         patch("rag.save_turn", new_callable=AsyncMock), \
         patch("rag.get_setting", return_value="off"):  # hyde_enabled=off
        await rag_query(MagicMock(), "pregunta", "ns", "user1", expertise_area="médico", tenant=mock_tenant)

    mock_hyde.assert_not_called()


@pytest.mark.asyncio
async def test_hyde_enabled_uses_hypothetical_for_search():
    """When HyDE returns a result, retrieve_context must receive the hypothetical, not original."""
    from rag import rag_query
    mock_tenant = MagicMock()
    mock_tenant.web_search_enabled = False
    mock_tenant.language_code = None

    captured_query = {}

    async def mock_retrieve(db, query, namespace):
        captured_query["q"] = query
        return []

    with patch("rag.get_history", new_callable=AsyncMock, return_value=[]), \
         patch("rag._reformulate_query", new_callable=AsyncMock, return_value="pregunta original"), \
         patch("rag._hyde_query", new_callable=AsyncMock, return_value="respuesta hipotética"), \
         patch("rag.retrieve_context", side_effect=mock_retrieve), \
         patch("rag._triage_response", new_callable=AsyncMock, return_value=("off_topic", "lo siento")), \
         patch("rag.save_turn", new_callable=AsyncMock), \
         patch("rag.get_setting", return_value="on"):
        await rag_query(MagicMock(), "pregunta original", "ns", "user1", expertise_area="médico", tenant=mock_tenant)

    assert captured_query["q"] == "respuesta hipotética"


# ─── Unit: contextual retrieval / index_chunks (T6) ──────────────────────────

@pytest.mark.asyncio
async def test_index_chunks_no_full_doc_skips_contextual():
    """FAQ chunks (full_doc_text=None) must not call _add_contextual_summary."""
    from rag import index_chunks
    mock_db = MagicMock()
    mock_db.add_all = MagicMock()
    mock_db.commit = AsyncMock()

    with patch("rag.call_embeddings", new_callable=AsyncMock, return_value=[[0.1] * 5]) as mock_emb, \
         patch("rag._add_contextual_summary", new_callable=AsyncMock) as mock_ctx, \
         patch("rag.get_setting", return_value="gpt-4o-mini"):
        await index_chunks(mock_db, [{"content": "texto", "source": "doc.pdf", "page": 1}], "ns")

    mock_ctx.assert_not_called()
    # embedding input must be original content, not contextual
    mock_emb.assert_called_once_with(["texto"])


@pytest.mark.asyncio
async def test_index_chunks_with_full_doc_calls_contextual():
    """When full_doc_text is provided and llm_context_model is set, contextual summary is prepended."""
    from rag import index_chunks
    mock_db = MagicMock()
    mock_db.add_all = MagicMock()
    mock_db.commit = AsyncMock()

    with patch("rag.call_embeddings", new_callable=AsyncMock, return_value=[[0.1] * 5]) as mock_emb, \
         patch("rag._add_contextual_summary", new_callable=AsyncMock, return_value="contexto del doc") as mock_ctx, \
         patch("rag.get_setting", return_value="gpt-4o-mini"):
        await index_chunks(
            mock_db,
            [{"content": "texto del chunk", "source": "doc.pdf", "page": 1}],
            "ns",
            full_doc_text="documento completo",
        )

    mock_ctx.assert_called_once_with("documento completo", "texto del chunk")
    # embedding receives context + chunk, not just chunk
    mock_emb.assert_called_once_with(["contexto del doc\ntexto del chunk"])


@pytest.mark.asyncio
async def test_index_chunks_contextual_failure_falls_back():
    """If _add_contextual_summary returns '', embed the chunk without context."""
    from rag import index_chunks
    mock_db = MagicMock()
    mock_db.add_all = MagicMock()
    mock_db.commit = AsyncMock()

    with patch("rag.call_embeddings", new_callable=AsyncMock, return_value=[[0.1] * 5]) as mock_emb, \
         patch("rag._add_contextual_summary", new_callable=AsyncMock, return_value="") as mock_ctx, \
         patch("rag.get_setting", return_value="gpt-4o-mini"):
        await index_chunks(
            mock_db,
            [{"content": "texto", "source": "doc.pdf", "page": 1}],
            "ns",
            full_doc_text="doc",
        )

    mock_ctx.assert_called_once()
    # fallback: embed original content only
    mock_emb.assert_called_once_with(["texto"])


@pytest.mark.asyncio
async def test_index_chunks_stores_original_content_not_contextual():
    """DB must always store original chunk content, never the contextual text."""
    from rag import index_chunks
    from db import DocumentChunk
    mock_db = MagicMock()
    stored_chunks = []
    mock_db.add_all = MagicMock(side_effect=lambda chunks: stored_chunks.extend(chunks))
    mock_db.commit = AsyncMock()

    with patch("rag.call_embeddings", new_callable=AsyncMock, return_value=[[0.1] * 5]), \
         patch("rag._add_contextual_summary", new_callable=AsyncMock, return_value="CONTEXTO PREPENDED"), \
         patch("rag.get_setting", return_value="gpt-4o-mini"):
        await index_chunks(
            mock_db,
            [{"content": "original chunk content", "source": "doc.pdf", "page": 1}],
            "ns",
            full_doc_text="full doc",
        )

    assert len(stored_chunks) == 1
    assert stored_chunks[0].content == "original chunk content"
    assert "CONTEXTO" not in stored_chunks[0].content


@pytest.mark.asyncio
async def test_add_contextual_summary_no_model_returns_empty():
    """When LLM_CONTEXT_MODEL is not configured, skip contextual summary."""
    from rag import _add_contextual_summary
    with patch("rag.get_setting", return_value=""), \
         patch("rag.call_chat", new_callable=AsyncMock) as mock_chat:
        result = await _add_contextual_summary("doc completo", "chunk text")
    assert result == ""
    mock_chat.assert_not_called()


@pytest.mark.asyncio
async def test_add_contextual_summary_llm_failure_returns_empty():
    """LLM failure must return '' and not raise — upload must not fail."""
    from rag import _add_contextual_summary
    with patch("rag.get_setting", return_value="gpt-4o-mini"), \
         patch("rag.call_chat", new_callable=AsyncMock, side_effect=RuntimeError("rate limit")):
        result = await _add_contextual_summary("doc", "chunk")
    assert result == ""


def test_process_uploaded_file_returns_full_doc_text():
    """process_uploaded_file must return 3-tuple including full_doc_text."""
    from services.upload import process_uploaded_file
    content = b"Primera parte del documento.\n\nSegunda parte del documento."
    chunks, pages, full_text = process_uploaded_file(content, "doc.txt", "doc.txt")
    assert isinstance(chunks, list)
    assert pages == 1
    assert "Primera parte" in full_text
    assert "Segunda parte" in full_text
