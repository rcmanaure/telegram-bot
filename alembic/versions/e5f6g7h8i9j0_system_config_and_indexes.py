"""Add system_config table, conversation composite index, and tenant_id backfill

Revision ID: e5f6g7h8i9j0
Revises: d4e5f6g7h8i9
Create Date: 2026-06-01

- Create system_config table for runtime config overlay
- Add composite index on conversations(namespace, user_id, created_at)
- Backfill Conversation.tenant_id by JOINing tenants on slug=namespace
- Backfill UnansweredQuery.tenant_id by JOINing tenants on namespace=slug
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = 'e5f6g7h8i9j0'
down_revision: Union[str, Sequence[str], None] = 'd4e5f6g7h8i9'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create system_config table
    op.create_table(
        'system_config',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('key', sa.String(100), nullable=False),
        sa.Column('encrypted_value', sa.Text(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True),
                  server_default=sa.text('now()'),
                  onupdate=sa.text('now()')),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_system_config_key', 'system_config', ['key'], unique=True)

    # Add composite index on conversations(namespace, user_id, created_at)
    op.create_index(
        'ix_conversations_ns_user_created',
        'conversations',
        ['namespace', 'user_id', 'created_at'],
    )

    # Backfill Conversation.tenant_id by joining tenants on slug=namespace
    op.execute("""
        UPDATE conversations c
        SET tenant_id = t.id
        FROM tenants t
        WHERE c.namespace = t.slug AND c.tenant_id IS NULL
    """)

    # Backfill UnansweredQuery.tenant_id by joining tenants on namespace=slug
    op.execute("""
        UPDATE unanswered_queries uq
        SET tenant_id = t.id
        FROM tenants t
        WHERE uq.namespace = t.slug AND uq.tenant_id IS NULL
    """)


def downgrade() -> None:
    op.drop_index('ix_conversations_ns_user_created', table_name='conversations')
    op.drop_table('system_config')