"""WhatsApp multi-channel: WA columns on tenants + wa_service_windows table

Revision ID: b2c3d4e5f6g7
Revises: a1b2c3d4e5f6
Create Date: 2026-05-29 12:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = 'b2c3d4e5f6g7'
down_revision: Union[str, Sequence[str], None] = 'a1b2c3d4e5f6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # WhatsApp credential columns on tenants
    op.add_column('tenants', sa.Column('wa_phone_number_id', sa.String(length=100), nullable=True))
    op.add_column('tenants', sa.Column('wa_access_token', sa.String(length=500), nullable=True))
    op.add_column('tenants', sa.Column('wa_app_secret', sa.String(length=100), nullable=True))
    op.add_column('tenants', sa.Column('wa_business_id', sa.String(length=100), nullable=True))
    op.add_column('tenants', sa.Column('wa_verify_token', sa.String(length=100), nullable=True))
    op.add_column('tenants', sa.Column('wa_reengagement_template', sa.String(length=200), nullable=True))
    op.add_column('tenants', sa.Column('channels', sa.Text(), nullable=True, server_default='telegram'))

    # 24-hour service window tracker
    op.create_table(
        'wa_service_windows',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('tenant_id', sa.Integer(), sa.ForeignKey('tenants.id'), nullable=False),
        sa.Column('user_id', sa.String(length=100), nullable=False),
        sa.Column('last_user_message_at', sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('tenant_id', 'user_id', name='uq_wa_window_tenant_user'),
    )


def downgrade() -> None:
    op.drop_table('wa_service_windows')
    op.drop_column('tenants', 'channels')
    op.drop_column('tenants', 'wa_reengagement_template')
    op.drop_column('tenants', 'wa_verify_token')
    op.drop_column('tenants', 'wa_business_id')
    op.drop_column('tenants', 'wa_app_secret')
    op.drop_column('tenants', 'wa_access_token')
    op.drop_column('tenants', 'wa_phone_number_id')