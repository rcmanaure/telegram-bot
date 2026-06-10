"""Tests for RLS integration (requires Docker — integration tests).

Validates that:
- tenant_session(slug) sets the RLS GUC correctly
- Unscoped TenantSessionLocal sessions see zero rows (GUC not set)
- Admin engine sees all rows (bypasses RLS)
- Cross-tenant isolation: tenant-a cannot see tenant-b data
- SET LOCAL app.current_tenant composes with SET LOCAL hnsw.*
"""
import pytest

from sqlalchemy import text, select
from sqlalchemy.ext.asyncio import AsyncSession

from db import (
    AsyncSessionLocal,
    TenantSessionLocal,
    tenant_session,
    DocumentChunk,
)


# ─── Helpers ──────────────────────────────────────────────────────────────────

async def _insert_chunk(db: AsyncSession, namespace: str, source: str, content: str):
    """Insert a test chunk via admin session (bypasses RLS)."""
    chunk = DocumentChunk(
        namespace=namespace,
        source=source,
        page=1,
        content=content,
    )
    db.add(chunk)
    await db.flush()


async def _count_chunks(db: AsyncSession, namespace: str | None = None) -> int:
    """Count chunks visible in the current session, optionally filtered by namespace."""
    if namespace:
        result = await db.execute(
            text("SELECT COUNT(*) FROM document_chunks WHERE namespace = :ns"),
            {"ns": namespace},
        )
    else:
        result = await db.execute(text("SELECT COUNT(*) FROM document_chunks"))
    return result.scalar_one()


# ─── Integration tests ────────────────────────────────────────────────────────

@pytest.mark.integration
class TestRLSIsolation:
    """These tests require a running Postgres with RLS policies applied."""

    @pytest.fixture(autouse=True)
    async def _setup_test_data(self):
        """Insert test data via admin session, clean up after."""
        async with AsyncSessionLocal() as db:
            # Clean up any leftover test data
            await db.execute(
                text("DELETE FROM document_chunks WHERE namespace LIKE 'test-rls-%'")
            )
            await db.commit()

            # Insert chunks for two test tenants
            await _insert_chunk(db, "test-rls-tenant-a", "source-a", "Content A1")
            await _insert_chunk(db, "test-rls-tenant-a", "source-a2", "Content A2")
            await _insert_chunk(db, "test-rls-tenant-b", "source-b", "Content B1")
            await db.commit()

        yield  # test runs here

        async with AsyncSessionLocal() as db:
            await db.execute(
                text("DELETE FROM document_chunks WHERE namespace LIKE 'test-rls-%'")
            )
            await db.commit()

    async def test_admin_sees_all_tenants(self):
        """Admin engine (owner role) bypasses RLS — sees all rows."""
        async with AsyncSessionLocal() as db:
            count = await _count_chunks(db, "test-rls-tenant-a")
            assert count == 2
            count = await _count_chunks(db, "test-rls-tenant-b")
            assert count == 1

    async def test_tenant_session_scoped(self):
        """tenant_session(slug) sets RLS GUC — tenant sees only own rows."""
        async with tenant_session("test-rls-tenant-a") as db:
            count_a = await _count_chunks(db, "test-rls-tenant-a")
            count_b = await _count_chunks(db, "test-rls-tenant-b")
            assert count_a == 2
            assert count_b == 0  # RLS blocks cross-tenant access

    async def test_tenant_session_b_sees_own_data(self):
        """tenant-b sees only its own data."""
        async with tenant_session("test-rls-tenant-b") as db:
            count_a = await _count_chunks(db, "test-rls-tenant-a")
            count_b = await _count_chunks(db, "test-rls-tenant-b")
            assert count_a == 0
            assert count_b == 1

    async def test_unscoped_tenant_session_sees_zero(self):
        """TenantSessionLocal without GUC set sees zero rows."""
        async with TenantSessionLocal() as db:
            # No set_config called — GUC is empty, matches no namespace
            count = await _count_chunks(db)
            assert count == 0

    async def test_bot_still_answers_with_rls_enabled(self):
        """Regression: RLS on the webhook path does not break the live bot.

        This is the mandatory regression test from the eng review.
        Simulates what bot.py does: tenant_session(slug) → query chunks.
        """
        async with tenant_session("test-rls-tenant-a") as db:
            result = await db.execute(
                select(DocumentChunk.source, DocumentChunk.content)
                .where(DocumentChunk.namespace == "test-rls-tenant-a")
            )
            rows = result.fetchall()
            assert len(rows) == 2
            assert all(r.source in ("source-a", "source-a2") for r in rows)

    async def test_set_local_composes_with_hnsw(self):
        """SET LOCAL app.current_tenant + SET LOCAL hnsw.ef_search compose.

        Both SET LOCAL statements live in the same transaction scope.
        Verify the GUC is set AND the HNSW param can be set afterwards.
        """
        async with tenant_session("test-rls-tenant-a") as db:
            # Verify the tenant GUC is set
            result = await db.execute(
                text("SELECT current_setting('app.current_tenant', true)")
            )
            guc_val = result.scalar_one()
            assert guc_val == "test-rls-tenant-a"

            # Set HNSW params (same pattern as retrieve_context)
            await db.execute(text("SET LOCAL hnsw.ef_search = 40"))
            # This should not raise — confirms composability
            result = await db.execute(
                text("SELECT current_setting('hnsw.ef_search')")
            )
            assert result.scalar_one() == "40"

            # RLS still works after HNSW SET LOCAL
            count = await _count_chunks(db, "test-rls-tenant-a")
            assert count == 2