"""add chunk_type, metadata to DocumentChunk; doc_structure_summary to Tenant

Revision ID: b4c5d6e7f890
Revises: a1b2c3d4e5f6
Create Date: 2026-06-09

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = 'b4c5d6e7f890'
down_revision: Union[str, Sequence[str], None] = 'h8i9j0k1l2m3'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('document_chunks', sa.Column('chunk_type', sa.String(length=20), nullable=True))
    op.add_column('document_chunks', sa.Column('metadata', sa.JSON(), nullable=True, server_default='{}'))
    op.add_column('tenants', sa.Column('doc_structure_summary', sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column('tenants', 'doc_structure_summary')
    op.drop_column('document_chunks', 'metadata')
    op.drop_column('document_chunks', 'chunk_type')