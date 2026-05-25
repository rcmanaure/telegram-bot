"""
FastAPI backend:
- POST /upload   → ingest a PDF into pgvector
- GET  /health   → health check
- GET  /stats    → show indexed docs per namespace
"""
import io
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, text
from pypdf import PdfReader
from db import init_db, get_db, DocumentChunk
from rag import chunk_text, index_chunks
from config import settings

MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    print(f"🚀 RAG Bot API running on port {settings.app_port}")
    yield


app = FastAPI(
    title="RAG Bot API",
    description="Upload documents, query them via Telegram bot",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health():
    return {"status": "ok", "model": settings.llm_model}


@app.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    namespace: str = Form(default=None),
    db: AsyncSession = Depends(get_db),
):
    """
    Upload a PDF and index it for RAG queries.

    - **file**: PDF file to upload (max 10 MB)
    - **namespace**: logical grouping (e.g. company name). Defaults to DEFAULT_NAMESPACE.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported")

    namespace = namespace or settings.default_namespace
    content = await file.read()

    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "File too large. Maximum size is 10MB.")

    # Parse PDF
    try:
        reader = PdfReader(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(400, f"Could not read PDF: {e}")

    all_chunks = []
    for page_num, page in enumerate(reader.pages):
        page_text = page.extract_text() or ""
        if page_text.strip():
            chunks = chunk_text(page_text, source=file.filename, page=page_num + 1)
            all_chunks.extend(chunks)

    if not all_chunks:
        raise HTTPException(400, "No text could be extracted from this PDF")

    # Index chunks into pgvector
    stored = await index_chunks(db, all_chunks, namespace)

    return {
        "status": "indexed",
        "namespace": namespace,
        "filename": file.filename,
        "pages_processed": len(reader.pages),
        "chunks_stored": stored,
        "message": f"✅ Document ready. Users can now ask questions about '{file.filename}' on Telegram."
    }


@app.get("/stats")
async def stats(db: AsyncSession = Depends(get_db)):
    """Show how many chunks are indexed per namespace."""
    result = await db.execute(
        select(
            DocumentChunk.namespace,
            DocumentChunk.source,
            func.count(DocumentChunk.id).label("chunks"),
        )
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


@app.delete("/namespace/{namespace}")
async def delete_namespace(namespace: str, db: AsyncSession = Depends(get_db)):
    """Delete all documents in a namespace (for cleanup/demo resets)."""
    await db.execute(
        text("DELETE FROM document_chunks WHERE namespace = :ns"),
        {"ns": namespace}
    )
    await db.execute(
        text("DELETE FROM conversations WHERE namespace = :ns"),
        {"ns": namespace}
    )
    await db.commit()
    return {"status": "deleted", "namespace": namespace}
