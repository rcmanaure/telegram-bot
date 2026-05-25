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
    """Returns fallback message without making any API call."""
    from rag import generate_answer
    result = await generate_answer([], "What hours are you open?", [])
    assert "couldn't find" in result.lower()


# ─── Unit: index_chunks with empty list ──────────────────────────────────────

@pytest.mark.asyncio
async def test_index_chunks_empty():
    """Returns 0 immediately without touching DB or embeddings API."""
    from rag import index_chunks
    assert await index_chunks(None, [], "test_ns") == 0


# ─── API: upload validation (DB startup mocked — no live services needed) ────

@pytest.fixture
def api_client():
    from unittest.mock import AsyncMock, patch
    import main as main_module
    from fastapi.testclient import TestClient
    with patch("main.init_db", new_callable=AsyncMock):
        with TestClient(main_module.app) as client:
            yield client


def test_health_endpoint(api_client):
    r = api_client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_upload_rejects_non_pdf(api_client):
    r = api_client.post(
        "/upload",
        files={"file": ("readme.txt", b"some text", "text/plain")},
        data={"namespace": "test"},
    )
    assert r.status_code == 400
    assert "PDF" in r.json()["detail"]


def test_upload_accepts_uppercase_extension(api_client):
    r = api_client.post(
        "/upload",
        files={"file": ("DOCUMENT.PDF", b"fake content", "application/pdf")},
        data={"namespace": "test"},
    )
    # Extension check passes — error must NOT be the "Only PDF files" rejection
    assert r.json().get("detail") != "Only PDF files are supported"


def test_upload_rejects_oversized_file(api_client):
    oversized = b"0" * (11 * 1024 * 1024)
    r = api_client.post(
        "/upload",
        files={"file": ("big.pdf", oversized, "application/pdf")},
        data={"namespace": "test"},
    )
    assert r.status_code == 413


def test_upload_rejects_corrupted_pdf(api_client):
    r = api_client.post(
        "/upload",
        files={"file": ("bad.pdf", b"not-a-real-pdf", "application/pdf")},
        data={"namespace": "test"},
    )
    assert r.status_code == 400
    assert "Could not read PDF" in r.json()["detail"]


# ─── Integration: full pipeline ───────────────────────────────────────────────
# Requires: docker compose up -d && postgres + api healthy
# Run: docker compose exec api python -m pytest tests/ -v -m integration

@pytest.mark.integration
@pytest.mark.asyncio
async def test_embed_and_retrieve(tmp_path):
    """Upload one chunk, retrieve it, assert similarity > 0."""
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

        # cleanup
        from sqlalchemy import text
        await db.execute(text("DELETE FROM document_chunks WHERE namespace = '_smoke_test'"))
        await db.commit()
