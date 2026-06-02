"""Add feedback table for user thumbs up/down ratings

Revision ID: f6g7h8i9j0k1
Revises: e5f6g7h8i9j0
Create Date: 2026-06-02

- Create feedback table for storing user feedback on bot responses
- Index on (namespace, created_at) for dashboard queries
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = 'f6g7h8i9j0k1'
down_revision: Union[str, Sequence[str], None] = 'e5f6g7h8i9j0'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'feedback',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('tenant_id', sa.Integer(), sa.ForeignKey('tenants.id'), nullable=True),
        sa.Column('user_id', sa.String(64), nullable=False),
        sa.Column('namespace', sa.String(128), nullable=False),
        sa.Column('message_id', sa.String(64), nullable=True),
        sa.Column('rating', sa.String(16), nullable=False),
        sa.Column('comment', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_feedback_user_id', 'feedback', ['user_id'])
    op.create_index('ix_feedback_ns_created', 'feedback', ['namespace', 'created_at'])


def downgrade() -> None:
    op.drop_index('ix_feedback_ns_created', table_name='feedback')
    op.drop_index('ix_feedback_user_id', table_name='feedback')
    op.drop_table('feedback')