"""REST API routes — health, upload, stats, update_tenant, delete_namespace."""
import logging

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from pydantic import BaseModel
from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from config import settings
from config_overlay import get_setting
from db import get_db, DocumentChunk, Feedback, Tenant
from dependencies import require_tenant
from limiter import limiter
from rag import chunk_text, flush_tool_cache, index_chunks
from services.upload import MAX_UPLOAD_BYTES, _IMAGE_EXTS, describe_image_for_upload, normalize_source_name, process_uploaded_file
from state import get_app

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/health")
async def health():
    return {
        "status": "ok",
        "model": get_setting("llm_model", settings.llm_model),
        "fallback_model": get_setting("llm_fallback_model", settings.llm_fallback_model),
    }


@router.post("/upload")
@limiter.limit("10/minute")
async def upload_document(
    request: Request,
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
    tenant: Tenant = Depends(require_tenant),
):
    content = await file.read()
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "File too large. Maximum size is 10MB.")

    source_name = normalize_source_name(file.filename)
    fname_lower = file.filename.lower()
    full_doc_text: str | None = None
    if any(fname_lower.endswith(ext) for ext in _IMAGE_EXTS):
        description = await describe_image_for_upload(content, fname_lower)
        all_chunks = chunk_text(description, source=source_name, page=1)
        pages_processed = 1
    else:
        all_chunks, pages_processed, full_doc_text = process_uploaded_file(content, fname_lower, source_name)

    if not all_chunks:
        raise HTTPException(400, "No extractable text in file")

    # Upsert: delete old chunks for this source, then insert new ones atomically
    await db.execute(
        text("DELETE FROM document_chunks WHERE namespace = :ns AND source = :src"),
        {"ns": tenant.slug, "src": source_name},
    )
    stored = await index_chunks(db, all_chunks, tenant.slug, auto_commit=False, full_doc_text=full_doc_text)
    await db.commit()
    flush_tool_cache(tenant.slug)

    return {
        "status": "indexed",
        "namespace": tenant.slug,
        "filename": source_name,
        "pages_processed": pages_processed,
        "chunks_stored": stored,
    }


@router.get("/stats")
@limiter.limit("10/minute")
async def stats(
    request: Request,
    db: AsyncSession = Depends(get_db),
    tenant: Tenant = Depends(require_tenant),
):
    result = await db.execute(
        select(
            DocumentChunk.namespace,
            DocumentChunk.source,
            func.count(DocumentChunk.id).label("chunks"),
        )
        .where(DocumentChunk.namespace == tenant.slug)
        .group_by(DocumentChunk.namespace, DocumentChunk.source)
        .order_by(DocumentChunk.namespace)
    )
    rows = result.fetchall()
    return {
        "indexed_documents": [
            {"namespace": r.namespace, "source": r.source, "chunks": r.chunks}
            for r in rows
        ]
    }


class TenantUpdate(BaseModel):
    expertise_area: str


@router.patch("/tenant")
async def update_tenant(
    body: TenantUpdate,
    db: AsyncSession = Depends(get_db),
    tenant: Tenant = Depends(require_tenant),
):
    tenant.expertise_area = body.expertise_area
    db.add(tenant)
    await db.commit()
    await db.refresh(tenant)

    # Update the in-memory cached tenant so the bot uses the new value immediately
    tg_app = get_app(tenant.bot_token)
    if tg_app:
        tg_app.bot_data["tenant"] = tenant

    return {"status": "updated", "expertise_area": tenant.expertise_area}


@router.delete("/namespace")
async def delete_namespace(
    db: AsyncSession = Depends(get_db),
    tenant: Tenant = Depends(require_tenant),
):
    ns = tenant.slug
    await db.execute(text("DELETE FROM document_chunks WHERE namespace = :ns"), {"ns": ns})
    await db.execute(text("DELETE FROM conversations WHERE namespace = :ns"), {"ns": ns})
    await db.execute(text("DELETE FROM unanswered_queries WHERE namespace = :ns"), {"ns": ns})
    await db.execute(text("DELETE FROM feedback WHERE namespace = :ns"), {"ns": ns})
    await db.commit()
    return {"status": "deleted", "namespace": ns}


@router.get("/feedback")
async def get_feedback(
    request: Request,
    db: AsyncSession = Depends(get_db),
    tenant: Tenant = Depends(require_tenant),
    limit: int = 100,
):
    """Return recent feedback for the authenticated tenant's namespace."""
    result = await db.execute(
        select(Feedback)
        .where(Feedback.namespace == tenant.slug)
        .order_by(Feedback.created_at.desc())
        .limit(min(limit, 500))
    )
    rows = result.scalars().all()
    return {
        "feedback": [
            {
                "id": r.id,
                "user_id": r.user_id,
                "message_id": r.message_id,
                "rating": r.rating,
                "comment": r.comment,
                "created_at": r.created_at.isoformat() if r.created_at else None,
            }
            for r in rows
        ],
        "total": len(rows),
    }