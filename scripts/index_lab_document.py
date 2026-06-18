#!/usr/bin/env python
"""Index sp-diagnostico-histologico.md into a tenant's namespace.

Usage:
    python scripts/index_lab_document.py <tenant_slug> [--rebuild]

Example:
    docker compose exec api python scripts/index_lab_document.py demo-lab --rebuild
"""
import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sqlalchemy import select, text
from db import AsyncSessionLocal, DocumentChunk, Tenant
from rag import chunk_text, index_chunks


async def index_lab_document(tenant_slug: str, rebuild: bool = False):
    """Index sp-diagnostico-histologico.md into tenant namespace."""
    lab_doc_path = Path(__file__).parent.parent / "sp-diagnostico-histologico.md"

    if not lab_doc_path.exists():
        print(f"❌ Lab document not found: {lab_doc_path}")
        return False

    # Connect to DB
    async with AsyncSessionLocal() as db:
        # Check tenant exists
        tenant = await db.scalar(
            select(Tenant).where(Tenant.slug == tenant_slug)
        )
        if not tenant:
            print(f"❌ Tenant not found: {tenant_slug}")
            return False

        print(f"📋 Indexing lab document for tenant: {tenant_slug}")

        # Read and chunk document
        content = lab_doc_path.read_text(encoding="utf-8")
        chunks = chunk_text(content, source="sp-diagnostico-histologico")
        print(f"📦 Created {len(chunks)} chunks from document")

        # If --rebuild, delete existing chunks first
        if rebuild:
            deleted = await db.execute(
                text(
                    "DELETE FROM document_chunks WHERE namespace = :ns AND source = :src"
                ),
                {"ns": tenant_slug, "src": "sp-diagnostico-histologico"},
            )
            print(f"🗑️  Deleted {deleted.rowcount} existing chunks")

        # Index chunks
        stored = await index_chunks(
            db, chunks, tenant_slug, auto_commit=True
        )
        print(f"✅ Indexed {stored} chunks successfully")

        # Verify
        count = await db.scalar(
            select(len(
                await db.execute(
                    text(
                        "SELECT 1 FROM document_chunks WHERE namespace = :ns AND source = :src"
                    ),
                    {"ns": tenant_slug, "src": "sp-diagnostico-histologico"},
                )
            ))
        )
        print(f"✓ Total chunks in namespace: {stored}")

        return True


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    tenant_slug = sys.argv[1]
    rebuild = "--rebuild" in sys.argv

    try:
        success = asyncio.run(index_lab_document(tenant_slug, rebuild))
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
