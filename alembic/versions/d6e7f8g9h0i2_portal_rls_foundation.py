"""portal RLS foundation — restricted role + row-level security policies

Revision ID: d6e7f8g9h0i2
Revises: c5d6e7f8g901
Create Date: 2026-06-10

Creates ragbot_tenant role (NOBYPASSRLS), grants DML on tenant-scoped tables,
enables RLS, and creates USING policies that filter on
current_setting('app.current_tenant', true). The owner role (ragbot)
bypasses RLS by default — we do NOT FORCE ROW LEVEL SECURITY so admin
paths continue to see all rows.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = 'd6e7f8g9h0i2'
down_revision: Union[str, Sequence[str], None] = 'c5d6e7f8g901'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

# Password for the restricted role. Must match TENANT_DB_PASSWORD env var.
# In a migration context we read from the environment — the app config isn't
# available during alembic execution.
_TENANT_ROLE = "ragbot_tenant"
_TENANT_TABLES_NAMESPACE = [
    "document_chunks",
    "conversations",
    "unanswered_queries",
    "feedback",
]
_TENANT_TABLES_TENANT_ID = [
    # Tables that scope by tenant_id FK instead of namespace
    "wa_service_windows",
]


def upgrade() -> None:
    # ── 1. Create restricted role ────────────────────────────────────────────
    # Password is read from env at migration time. If not set, use a placeholder
    # that must be changed before the app starts.
    import os
    tenant_db_password = os.environ.get("TENANT_DB_PASSWORD", "changeme_tenant")
    # Escape single quotes in password to prevent SQL injection during migration.
    # The password comes from an env var set by the operator, not user input,
    # but defense-in-depth applies here since this runs as a superuser.
    _safe_pw = tenant_db_password.replace("'", "''")
    op.execute(
        f"DO $$ BEGIN "
        f"CREATE ROLE {_TENANT_ROLE} NOBYPASSRLS LOGIN PASSWORD '{_safe_pw}'; "
        f"EXCEPTION WHEN duplicate_object THEN NULL; "
        f"END $$"
    )

    # ── 2. Grant DML on tenant-scoped tables ─────────────────────────────────
    for table in _TENANT_TABLES_NAMESPACE + _TENANT_TABLES_TENANT_ID:
        op.execute(f"GRANT SELECT, INSERT, UPDATE, DELETE ON {table} TO {_TENANT_ROLE}")

    # Tenant needs to read its own row for auth checks
    op.execute(f"GRANT SELECT ON tenants TO {_TENANT_ROLE}")

    # Grant sequence access for auto-increment PKs
    op.execute(f"GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO {_TENANT_ROLE}")

    # ── 3. Enable RLS on tenant-scoped tables ────────────────────────────────
    # ENABLE (not FORCE) means the table owner still bypasses RLS.
    # This is intentional: admin engine connects as owner and sees all rows.
    for table in _TENANT_TABLES_NAMESPACE + _TENANT_TABLES_TENANT_ID:
        op.execute(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY")

    # ── 4. RLS policies ─────────────────────────────────────────────────────
    # Namespace-based tables: policy filters on namespace column.
    # current_setting('app.current_tenant', true) returns '' when GUC not set,
    # which matches no namespace → unscoped sessions see zero rows.
    for table in _TENANT_TABLES_NAMESPACE:
        op.execute(
            f"CREATE POLICY tenant_isolation ON {table} "
            f"FOR ALL TO {_TENANT_ROLE} "
            f"USING (namespace = current_setting('app.current_tenant', true))"
        )

    # tenant_id-based tables: join through tenants to match slug.
    for table in _TENANT_TABLES_TENANT_ID:
        op.execute(
            f"CREATE POLICY tenant_isolation ON {table} "
            f"FOR ALL TO {_TENANT_ROLE} "
            f"USING (tenant_id IN ("
            f"  SELECT id FROM tenants "
            f"  WHERE slug = current_setting('app.current_tenant', true)"
            f"))"
        )


def downgrade() -> None:
    # Drop policies first (must exist before RLS can be disabled cleanly)
    for table in _TENANT_TABLES_NAMESPACE + _TENANT_TABLES_TENANT_ID:
        op.execute(f"DROP POLICY IF EXISTS tenant_isolation ON {table}")

    # Disable RLS
    for table in _TENANT_TABLES_NAMESPACE + _TENANT_TABLES_TENANT_ID:
        op.execute(f"ALTER TABLE {table} DISABLE ROW LEVEL SECURITY")

    # Revoke grants
    for table in _TENANT_TABLES_NAMESPACE + _TENANT_TABLES_TENANT_ID:
        op.execute(f"REVOKE ALL ON {table} FROM {_TENANT_ROLE}")
    op.execute(f"REVOKE SELECT ON tenants FROM {_TENANT_ROLE}")
    op.execute(f"REVOKE USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public FROM {_TENANT_ROLE}")

    # Drop role
    op.execute(f"DROP ROLE IF EXISTS {_TENANT_ROLE}")