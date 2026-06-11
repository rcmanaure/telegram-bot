"""add indexes on chunk_type, convert metadata to JSONB, add GIN index

Revision ID: c5d6e7f8g901
Revises: b4c5d6e7f890
Create Date: 2026-06-09

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = 'c5d6e7f8g901'
down_revision: Union[str, Sequence[str], None] = 'b4c5d6e7f890'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Composite index on (namespace, chunk_type) — critical for retrieve_policy_chunks()
    # which filters WHERE namespace = :ns AND chunk_type IN (...). Without this,
    # every query with context triggers a sequential scan of all chunks in the namespace.
    op.create_index(
        'ix_document_chunks_namespace_type',
        'document_chunks',
        ['namespace', 'chunk_type'],
        if_not_exists=True,
    )

    # Convert metadata column from JSON to JSONB for GIN index support.
    # JSONB is the standard type for JSON columns that need indexing in PostgreSQL.
    # The USING clause converts existing JSON values in-place.
    op.execute(
        "ALTER TABLE document_chunks "
        "ALTER COLUMN metadata TYPE jsonb USING metadata::jsonb"
    )

    # GIN index on metadata JSONB — used by retrieve_section_siblings() which filters
    # metadata->>'section_name'. GIN with jsonb_path_ops is optimal for containment
    # and key-path extraction queries.
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_document_chunks_metadata_gin "
        "ON document_chunks USING gin (metadata jsonb_path_ops)"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_document_chunks_metadata_gin")
    # Revert metadata column back to JSON
    op.execute(
        "ALTER TABLE document_chunks "
        "ALTER COLUMN metadata TYPE json USING metadata::json"
    )
    op.drop_index('ix_document_chunks_namespace_type', table_name='document_chunks', if_exists=True)