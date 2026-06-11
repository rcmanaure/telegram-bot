"""tenants RLS policy — restrict ragbot_tenant to own row

Revision ID: f8g9h0i1j2k3
Revises: e7f8g9h0i1j3
Create Date: 2026-06-10

Adds:
- ENABLE ROW LEVEL SECURITY on tenants table
- RLS policy: tenants can only see their own row (slug = app.current_tenant)

Defense-in-depth: even if future code queries tenants through a tenant-scoped
session, only the tenant's own row is visible. The admin role (ragbot) bypasses
RLS and continues to see all rows.

Note: SELECT on tenants is already granted to ragbot_tenant by the
portal_rls_foundation migration (d6e7f8g9h0i2). This migration only adds
the RLS policy; the GRANT is not duplicated here.
"""
from typing import Sequence, Union

from alembic import op


revision: str = 'f8g9h0i1j2k3'
down_revision: Union[str, Sequence[str], None] = 'e7f8g9h0i1j3'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_TENANT_ROLE = "ragbot_tenant"


def upgrade() -> None:
    # Enable RLS — owner role (ragbot) bypasses; only ragbot_tenant is restricted
    op.execute("ALTER TABLE tenants ENABLE ROW LEVEL SECURITY")

    # Policy: ragbot_tenant can only see the row matching their GUC slug
    op.execute(
        f"CREATE POLICY tenant_self ON tenants "
        f"FOR SELECT TO {_TENANT_ROLE} "
        f"USING (slug = current_setting('app.current_tenant', true))"
    )


def downgrade() -> None:
    op.execute(f"DROP POLICY IF EXISTS tenant_self ON tenants")
    op.execute("ALTER TABLE tenants DISABLE ROW LEVEL SECURITY")