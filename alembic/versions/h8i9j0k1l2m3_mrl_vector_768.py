"""MRL vector dimension: vector(512) -> vector(768)

Revision ID: h8i9j0k1l2m3
Revises: g7h8i9j0k1l2
Create Date: 2026-06-03

Upgrades embedding column from 512-dim to 768-dim MRL truncation of
text-embedding-3-small. 768-dim yields ~97% of full model accuracy vs
~92% at 512-dim, at ~8% additional embedding latency. Optimal for
mixed-domain tenants (medical labs, restaurants, pharmacies, gyms)
where Spanish-language query precision matters more than storage size.

Also bumps chunk_overlap 77→128 (config only — no migration needed) and
top_k_results 6→10 (config only) to improve recall across all tenants.

BEFORE deploying:
  1. pg_dump snapshot (this migration clears all embeddings)
  2. Ensure EMBEDDING_DIM=768 is set (or removed — 768 is the new default)

AFTER deploying:
  1. Re-upload all documents for every tenant via admin UI
  2. validate_config() at startup logs: "Embedding dimension check passed: 768-dim"

Storage vs 512: +50% per row (3KB → 4.5KB per chunk). Negligible for typical corpus sizes.
Storage vs original 1536: still -50% reduction.
"""
from typing import Sequence, Union

from alembic import op

revision: str = 'h8i9j0k1l2m3'
down_revision: Union[str, Sequence[str], None] = 'g7h8i9j0k1l2'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_embedding_hnsw")
    op.execute("UPDATE document_chunks SET embedding = NULL")
    op.execute(
        "ALTER TABLE document_chunks ALTER COLUMN embedding TYPE vector(768)"
    )
    op.execute(
        "CREATE INDEX ix_embedding_hnsw ON document_chunks "
        "USING hnsw (embedding vector_cosine_ops) "
        "WITH (m = 16, ef_construction = 128)"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_embedding_hnsw")
    op.execute("UPDATE document_chunks SET embedding = NULL")
    op.execute(
        "ALTER TABLE document_chunks ALTER COLUMN embedding TYPE vector(512)"
    )
    op.execute(
        "CREATE INDEX ix_embedding_hnsw ON document_chunks "
        "USING hnsw (embedding vector_cosine_ops) "
        "WITH (m = 16, ef_construction = 128)"
    )
