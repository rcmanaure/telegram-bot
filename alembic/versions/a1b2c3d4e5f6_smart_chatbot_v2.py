"""smart chatbot v2: contact_url, example_questions, operator_chat_id, unanswered_queries

Revision ID: a1b2c3d4e5f6
Revises: 3f8a1c2e9d47
Create Date: 2026-05-26 12:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = 'a1b2c3d4e5f6'
down_revision: Union[str, Sequence[str], None] = '3f8a1c2e9d47'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('tenants', sa.Column('contact_url', sa.String(length=512), nullable=True))
    op.add_column('tenants', sa.Column('example_questions', sa.JSON(), nullable=True))
    op.add_column('tenants', sa.Column('operator_chat_id', sa.String(length=64), nullable=True))

    op.create_table(
        'unanswered_queries',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('tenant_id', sa.Integer(), sa.ForeignKey('tenants.id'), nullable=True),
        sa.Column('namespace', sa.String(length=128), nullable=False),
        sa.Column('question', sa.Text(), nullable=False),
        sa.Column('user_id', sa.String(length=64), nullable=False),
        sa.Column('intent_category', sa.String(length=64), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index(
        'ix_unanswered_queries_tenant_created',
        'unanswered_queries',
        ['tenant_id', 'created_at'],
    )
    op.create_index(
        'ix_unanswered_queries_created_at',
        'unanswered_queries',
        ['created_at'],
    )


def downgrade() -> None:
    op.drop_index('ix_unanswered_queries_created_at', table_name='unanswered_queries')
    op.drop_index('ix_unanswered_queries_tenant_created', table_name='unanswered_queries')
    op.drop_table('unanswered_queries')
    op.drop_column('tenants', 'operator_chat_id')
    op.drop_column('tenants', 'example_questions')
    op.drop_column('tenants', 'contact_url')
