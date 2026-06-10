"""portal schema — portal_password_hash, TenantAuditLog, TenantUsage

Revision ID: e7f8g9h0i1j3
Revises: d6e7f8g9h0i2
Create Date: 2026-06-10

Adds:
- Tenant.portal_password_hash (nullable, bcrypt)
- TenantAuditLog table (knowledge mutation audit trail)
- TenantUsage table (monthly metric counters for E2 metering)

Grants RLS access on new tables to ragbot_tenant.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = 'e7f8g9h0i1j3'
down_revision: Union[str, Sequence[str], None] = 'd6e7f8g9h0i2'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_TENANT_ROLE = "ragbot_tenant"


def upgrade() -> None:
    # ── 1. portal_password_hash on tenants ────────────────────────────────────
    op.add_column(
        'tenants',
        sa.Column('portal_password_hash', sa.String(128), nullable=True),
    )

    # ── 2. TenantAuditLog ─────────────────────────────────────────────────────
    op.create_table(
        'tenant_audit_log',
        sa.Column('id', sa.Integer, primary_key=True, autoincrement=True),
        sa.Column('tenant_id', sa.Integer, sa.ForeignKey('tenants.id'), nullable=False, index=True),
        sa.Column('namespace', sa.String(128), nullable=False),
        sa.Column('actor', sa.String(64), nullable=False),
        sa.Column('action', sa.String(64), nullable=False),
        sa.Column('detail', sa.JSON, nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index('ix_audit_tenant_created', 'tenant_audit_log', ['tenant_id', 'created_at'])
    op.create_index('ix_audit_namespace_created', 'tenant_audit_log', ['namespace', 'created_at'])

    # ── 3. TenantUsage ────────────────────────────────────────────────────────
    op.create_table(
        'tenant_usage',
        sa.Column('id', sa.Integer, primary_key=True, autoincrement=True),
        sa.Column('tenant_id', sa.Integer, sa.ForeignKey('tenants.id'), nullable=False),
        sa.Column('period_year', sa.Integer, nullable=False),
        sa.Column('period_month', sa.Integer, nullable=False),
        sa.Column('metric', sa.String(64), nullable=False),
        sa.Column('value', sa.Integer, nullable=False, server_default='0'),
        sa.UniqueConstraint(
            'tenant_id', 'period_year', 'period_month', 'metric',
            name='uq_tenant_usage_period_metric',
        ),
    )
    op.create_index('ix_tenant_usage_period', 'tenant_usage', ['tenant_id', 'period_year', 'period_month'])

    # ── 4. RLS on new tables ──────────────────────────────────────────────────
    op.execute(f"GRANT SELECT, INSERT, UPDATE, DELETE ON tenant_audit_log TO {_TENANT_ROLE}")
    op.execute(f"GRANT SELECT, INSERT, UPDATE, DELETE ON tenant_usage TO {_TENANT_ROLE}")
    op.execute(f"GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO {_TENANT_ROLE}")

    op.execute("ALTER TABLE tenant_audit_log ENABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE tenant_usage ENABLE ROW LEVEL SECURITY")

    # Audit log: namespace-based (same as document_chunks, conversations, etc.)
    op.execute(
        f"CREATE POLICY tenant_isolation ON tenant_audit_log "
        f"FOR ALL TO {_TENANT_ROLE} "
        f"USING (namespace = current_setting('app.current_tenant', true))"
    )

    # Usage: tenant_id-based (same as wa_service_windows)
    op.execute(
        f"CREATE POLICY tenant_isolation ON tenant_usage "
        f"FOR ALL TO {_TENANT_ROLE} "
        f"USING (tenant_id IN ("
        f"  SELECT id FROM tenants "
        f"  WHERE slug = current_setting('app.current_tenant', true)"
        f"))"
    )


def downgrade() -> None:
    # Drop RLS policies + disable RLS
    op.execute("DROP POLICY IF EXISTS tenant_isolation ON tenant_usage")
    op.execute("DROP POLICY IF EXISTS tenant_isolation ON tenant_audit_log")
    op.execute("ALTER TABLE tenant_usage DISABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE tenant_audit_log DISABLE ROW LEVEL SECURITY")

    # Revoke grants
    op.execute(f"REVOKE ALL ON tenant_usage FROM {_TENANT_ROLE}")
    op.execute(f"REVOKE ALL ON tenant_audit_log FROM {_TENANT_ROLE}")

    # Drop tables
    op.drop_index('ix_tenant_usage_period', table_name='tenant_usage')
    op.drop_table('tenant_usage')
    op.drop_index('ix_audit_namespace_created', table_name='tenant_audit_log')
    op.drop_index('ix_audit_tenant_created', table_name='tenant_audit_log')
    op.drop_table('tenant_audit_log')

    # Drop column
    op.drop_column('tenants', 'portal_password_hash')