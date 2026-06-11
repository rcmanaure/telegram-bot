"""tenants RLS policy — restrict ragbot_tenant to own row

Revision ID: f8g9h0i1j2k3
Revises: e7f8g9h0i1j3
Create Date: 2026-06-10

Adds:
- ENABLE ROW LEVEL SECURITY on tenants table
- GRANT SELECT on tenants to ragbot_tenant
- RLS policy: tenants can only see their own row (slug = app.current_tenant)

Defense-in-depth: even if future code queries tenants through a tenant-scoped
session, only the tenant's own row is visible. The admin role (ragbot) bypasses
RLS and continues to see all rows.
"""
from typing import Sequence, Union

from alembic import op


revision: str = 'f8g9h0i1j2k3'
down_revision: Union[str, Sequence[str], None] = 'e7f8g9h0i1j3'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_TENANT_ROLE = "ragbot_tenant"


def upgrade() -> None:
    # Grant SELECT so ragbot_tenant can query the tenants table at all
    op.execute(f"GRANT SELECT ON tenants TO {_TENANT_ROLE}")

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
    op.execute(f"REVOKE SELECT ON tenants FROM {_TENANT_ROLE}")