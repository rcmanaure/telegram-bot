"""add expertise_area to tenants

Revision ID: 3f8a1c2e9d47
Revises: 766462df7dd1
Create Date: 2026-05-25 23:50:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = '3f8a1c2e9d47'
down_revision: Union[str, Sequence[str], None] = '766462df7dd1'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('tenants', sa.Column('expertise_area', sa.String(length=255), nullable=True, server_default=''))


def downgrade() -> None:
    op.drop_column('tenants', 'expertise_area')
