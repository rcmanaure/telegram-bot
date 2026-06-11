"""Usage metering and audit log helpers for E2/E6.

increment_usage / get_usage: per-tenant monthly counters (atomic upsert).
check_and_increment_usage: atomic limit check + increment (prevents TOCTOU).
write_audit_log: knowledge mutation audit trail for TenantAuditLog.
"""
import logging
from datetime import datetime, timezone

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from db import TenantAuditLog

logger = logging.getLogger(__name__)


async def increment_usage(
    db: AsyncSession,
    tenant_id: int,
    metric: str,
    delta: int = 1,
    *,
    auto_commit: bool = True,
) -> None:
    """Atomically increment a usage counter for the current month.

    Uses INSERT ... ON CONFLICT DO UPDATE so concurrent requests don't
    create duplicate rows or lose increments.
    """
    now = datetime.now(timezone.utc)
    await db.execute(
        text("""
            INSERT INTO tenant_usage (tenant_id, period_year, period_month, metric, value)
            VALUES (:tid, :year, :month, :metric, :delta)
            ON CONFLICT (tenant_id, period_year, period_month, metric)
            DO UPDATE SET value = tenant_usage.value + :delta
        """),
        {
            "tid": tenant_id,
            "year": now.year,
            "month": now.month,
            "metric": metric,
            "delta": delta,
        },
    )
    if auto_commit:
        await db.commit()


async def check_and_increment_usage(
    db: AsyncSession,
    tenant_id: int,
    metric: str,
    limit: int,
    *,
    auto_commit: bool = True,
) -> bool:
    """Atomically check the monthly limit and increment if under it.

    Prevents TOCTOU races where concurrent requests both pass get_usage()
    before either increments. Uses a single SQL statement that only
    increments if the current value is below the limit.

    Returns True if the increment succeeded (under limit), False if over limit.
    """
    now = datetime.now(timezone.utc)
    # Upsert the row, but ONLY increment if current value < limit.
    # The RETURNING clause gives us the new value so we can confirm.
    result = await db.execute(
        text("""
            INSERT INTO tenant_usage (tenant_id, period_year, period_month, metric, value)
            VALUES (:tid, :year, :month, :metric, 1)
            ON CONFLICT (tenant_id, period_year, period_month, metric)
            DO UPDATE SET value = tenant_usage.value + 1
            WHERE tenant_usage.value < :limit
            RETURNING value
        """),
        {
            "tid": tenant_id,
            "year": now.year,
            "month": now.month,
            "metric": metric,
            "limit": limit,
        },
    )
    row = result.scalar_one_or_none()
    if auto_commit:
        await db.commit()
    return row is not None


async def get_usage(
    db: AsyncSession,
    tenant_id: int,
    metric: str,
) -> int:
    """Return current-month usage for a metric. Returns 0 if no row exists."""
    now = datetime.now(timezone.utc)
    result = await db.execute(
        text("""
            SELECT value FROM tenant_usage
            WHERE tenant_id = :tid
              AND period_year = :year
              AND period_month = :month
              AND metric = :metric
        """),
        {"tid": tenant_id, "year": now.year, "month": now.month, "metric": metric},
    )
    row = result.scalar_one_or_none()
    return row if row is not None else 0


async def write_audit_log(
    db: AsyncSession,
    tenant_id: int,
    namespace: str,
    actor: str,
    action: str,
    detail: dict | None = None,
    *,
    auto_commit: bool = True,
) -> None:
    """Write an audit log entry for a knowledge mutation (E6).

    Actor format: "tenant:{slug}" for portal actions, "admin" for operator actions.
    Actions: "document.upload", "document.delete", "portal.query", "portal.login", etc.
    """
    db.add(TenantAuditLog(
        tenant_id=tenant_id,
        namespace=namespace,
        actor=actor,
        action=action,
        detail=detail,
    ))
    if auto_commit:
        await db.commit()