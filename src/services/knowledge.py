"""Shared knowledge-base CRUD service.

Single source of truth for document upsert / delete / list / FAQ-sync.
Called by the operator `/admin` routes today and by the tenant self-service
portal (PR2). Keeping this logic here (not in routes/admin.py) means both
surfaces upsert chunks the same way — same idempotent delete-then-insert,
same cache flush, same FAQ re-sync.

All functions take a namespace (== tenant.slug) explicitly. The caller is
responsible for authorization (operator admin vs require_tenant_session) and
for opening the correct DB session — under RLS the tenant-scoped session
already constrains writes to `app.current_tenant`, but every query here also
filters `namespace` explicitly for defense-in-depth and planner clarity.
"""
import logging

from fastapi import HTTPException
from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from db import DocumentChunk
from rag import chunk_text, flush_tool_cache, get_index_status, index_chunks, set_index_status, sync_faq_chunks
from services.upload import (
    normalize_source_name,
    process_upload_async,
)

logger = logging.getLogger(__name__)

# Re-export so callers have one import surface for FAQ sync.
sync_faq = sync_faq_chunks


async def process_upload(content: bytes, filename: str) -> tuple[list[dict], int, str | None]:
    """Turn raw uploaded bytes into indexable chunks.

    Delegates to process_upload_async which offloads CPU-bound PDF parsing
    to a thread pool, keeping the event loop free for other requests.
    Returns (chunks, pages_processed, full_doc_text).
    """
    return await process_upload_async(content, filename)


async def upsert_source(
    db: AsyncSession,
    namespace: str,
    source: str,
    chunks: list[dict],
    *,
    full_doc_text: str | None = None,
    auto_commit: bool = True,
) -> int:
    """Idempotently replace all chunks for (namespace, source).

    Deletes existing rows for the source then inserts the new chunks in the
    same transaction (uploads stay idempotent). Flushes the tool cache for the
    namespace so stale retrieval results are not served. Returns chunk count.
    """
    source = normalize_source_name(source)
    set_index_status(namespace, source, "procesando")
    await db.execute(
        text("DELETE FROM document_chunks WHERE namespace = :ns AND source = :src"),
        {"ns": namespace, "src": source},
    )
    stored = await index_chunks(
        db, chunks, namespace, auto_commit=False, full_doc_text=full_doc_text
    )
    if auto_commit:
        await db.commit()
    flush_tool_cache(namespace)
    set_index_status(namespace, source, "activo")
    logger.info("knowledge upsert ns=%s source=%s chunks=%d", namespace, source, stored)
    return stored


async def delete_source(
    db: AsyncSession, namespace: str, source: str, *, auto_commit: bool = True
) -> None:
    """Delete all chunks for a single (namespace, source)."""
    await db.execute(
        text("DELETE FROM document_chunks WHERE namespace = :ns AND source = :src"),
        {"ns": namespace, "src": normalize_source_name(source)},
    )
    if auto_commit:
        await db.commit()
    flush_tool_cache(namespace)
    logger.info("knowledge delete_source ns=%s source=%s", namespace, source)


async def delete_all(
    db: AsyncSession, namespace: str, tenant_id: int, *, auto_commit: bool = True
) -> None:
    """Purge all knowledge + conversation state for a tenant namespace.

    Mirrors the operator "delete docs" action: chunks, conversations,
    unanswered queries, feedback (by namespace) and WA service windows (by
    tenant_id).
    """
    await db.execute(text("DELETE FROM document_chunks WHERE namespace = :ns"), {"ns": namespace})
    await db.execute(text("DELETE FROM conversations WHERE namespace = :ns"), {"ns": namespace})
    await db.execute(text("DELETE FROM unanswered_queries WHERE namespace = :ns"), {"ns": namespace})
    await db.execute(text("DELETE FROM feedback WHERE namespace = :ns"), {"ns": namespace})
    await db.execute(text("DELETE FROM wa_service_windows WHERE tenant_id = :tid"), {"tid": tenant_id})
    if auto_commit:
        await db.commit()
    flush_tool_cache(namespace)
    logger.info("knowledge delete_all ns=%s", namespace)


async def list_sources(db: AsyncSession, namespace: str) -> list[dict]:
    """Return [{source, chunks}, ...] for a namespace, ordered by source."""
    result = await db.execute(
        select(DocumentChunk.source, func.count(DocumentChunk.id).label("chunks"))
        .where(DocumentChunk.namespace == namespace)
        .group_by(DocumentChunk.source)
        .order_by(DocumentChunk.source)
    )
    return [{"source": r.source, "chunks": r.chunks} for r in result.fetchall()]
