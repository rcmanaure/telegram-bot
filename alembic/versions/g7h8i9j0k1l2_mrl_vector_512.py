"""MRL vector dimension: vector(1536) -> vector(512)

Revision ID: g7h8i9j0k1l2
Revises: f6g7h8i9j0k1
Create Date: 2026-06-02

Switches embedding column to 512-dim to use MRL (Matryoshka Representation
Learning) truncation supported by text-embedding-3-small via OpenRouter.

BEFORE deploying:
  1. Take a pg_dump snapshot (this migration clears all embeddings)
  2. Verify MRL support: python scripts/test_mrl_dimensions.py

AFTER deploying:
  1. Confirm SELECT extversion FROM pg_extension WHERE extname = 'vector'
  2. Re-upload all documents via admin UI (embeddings were cleared)
  3. validate_config() at startup logs: "Embedding dimension check passed: 512-dim"

Storage savings: ~67% reduction vs vector(1536) float32.
"""
from typing import Sequence, Union

from alembic import op

revision: str = 'g7h8i9j0k1l2'
down_revision: Union[str, Sequence[str], None] = 'f6g7h8i9j0k1'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Drop existing HNSW index (must drop before ALTER COLUMN)
    op.execute("DROP INDEX IF EXISTS ix_embedding_hnsw")

    # Clear existing embeddings — they are 1536-dim and incompatible with
    # the new 512-dim column. Re-upload all documents after deploying.
    op.execute("UPDATE document_chunks SET embedding = NULL")

    # Change column type to 512-dim
    op.execute(
        "ALTER TABLE document_chunks ALTER COLUMN embedding TYPE vector(512)"
    )

    # Recreate HNSW index with ef_construction=128 (vs 64 previously)
    # HNSW does not support CONCURRENTLY — brief table lock during build.
    # Estimated < 1 min for < 10K rows (most of which are now NULL, so instant).
    op.execute(
        "CREATE INDEX ix_embedding_hnsw ON document_chunks "
        "USING hnsw (embedding vector_cosine_ops) "
        "WITH (m = 16, ef_construction = 128)"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_embedding_hnsw")
    op.execute("UPDATE document_chunks SET embedding = NULL")
    op.execute(
        "ALTER TABLE document_chunks ALTER COLUMN embedding TYPE vector(1536)"
    )
    op.execute(
        "CREATE INDEX ix_embedding_hnsw ON document_chunks "
        "USING hnsw (embedding vector_cosine_ops) "
        "WITH (m = 16, ef_construction = 64)"
    )
