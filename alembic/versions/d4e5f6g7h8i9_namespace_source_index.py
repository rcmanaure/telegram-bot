"""add compound index on document_chunks(namespace, source)

Supports the upsert pattern: DELETE WHERE namespace = :ns AND source = :src
before re-inserting chunks for a given document.

Revision ID: d4e5f6g7h8i9
Revises: c3d4e5f6g7h8
Create Date: 2026-06-01

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = 'd4e5f6g7h8i9'
down_revision: Union[str, Sequence[str], None] = 'c3d4e5f6g7h8'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_index(
        'ix_document_chunks_namespace_source',
        'document_chunks',
        ['namespace', 'source'],
    )


def downgrade() -> None:
    op.drop_index('ix_document_chunks_namespace_source', table_name='document_chunks')