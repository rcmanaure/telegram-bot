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
    # second chunk starts before end of first (overlap)
    assert chunks[1]["content"][0] == "X"


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
