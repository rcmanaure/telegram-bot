"""multi_tenant_schema

Revision ID: 766462df7dd1
Revises: af07e4f0ea14
Create Date: 2026-05-25 18:31:41.600287

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '766462df7dd1'
down_revision: Union[str, Sequence[str], None] = 'af07e4f0ea14'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. tenants table
    op.create_table(
        'tenants',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('slug', sa.String(length=64), nullable=False),
        sa.Column('api_key_hash', sa.String(length=64), nullable=False),
        sa.Column('webhook_secret', sa.String(length=64), nullable=False),
        sa.Column('bot_token', sa.String(length=128), nullable=False),
        sa.Column('plan', sa.String(length=32), nullable=True),
        sa.Column('billing_id', sa.String(length=128), nullable=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('now()')),
        sa.Column('active', sa.Boolean(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('api_key_hash'),
        sa.UniqueConstraint('bot_token'),
        sa.UniqueConstraint('slug'),
    )
    op.create_index('ix_tenants_slug', 'tenants', ['slug'])

    # 2. conversations: rename telegram_user_id → user_id, add channel + tenant_id
    op.add_column('conversations', sa.Column('user_id', sa.String(length=64), nullable=True))
    op.add_column('conversations', sa.Column('channel', sa.String(length=20), server_default='telegram'))
    op.add_column('conversations', sa.Column('tenant_id', sa.Integer(), nullable=True))

    # Copy data to new column
    op.execute("UPDATE conversations SET user_id = telegram_user_id")
    op.alter_column('conversations', 'user_id', nullable=False)

    op.create_foreign_key(
        'fk_conversations_tenant_id', 'conversations',
        'tenants', ['tenant_id'], ['id'],
    )
    op.create_index('ix_conversations_user_id', 'conversations', ['user_id'])
    op.drop_index('ix_conversations_telegram_user_id', table_name='conversations')
    op.drop_column('conversations', 'telegram_user_id')

    # 3. HNSW index on document_chunks.embedding
    # No CONCURRENTLY: migration runs at startup before traffic, table is empty,
    # lock is held for milliseconds. CONCURRENTLY is incompatible with transactions.
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_embedding_hnsw "
        "ON document_chunks USING hnsw (embedding vector_cosine_ops) "
        "WITH (m = 16, ef_construction = 64)"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_embedding_hnsw")

    op.add_column('conversations', sa.Column('telegram_user_id', sa.String(length=50), nullable=True))
    op.execute("UPDATE conversations SET telegram_user_id = user_id")
    op.alter_column('conversations', 'telegram_user_id', nullable=False)
    op.create_index('ix_conversations_telegram_user_id', 'conversations', ['telegram_user_id'])

    op.drop_constraint('fk_conversations_tenant_id', 'conversations', type_='foreignkey')
    op.drop_index('ix_conversations_user_id', table_name='conversations')
    op.drop_column('conversations', 'tenant_id')
    op.drop_column('conversations', 'channel')
    op.drop_column('conversations', 'user_id')

    op.drop_index('ix_tenants_slug', table_name='tenants')
    op.drop_table('tenants')
