"""Usage metering helpers for E2 (per-tenant monthly counters).

Uses INSERT ... ON CONFLICT DO UPDATE for atomic upsert.
Monthly reset is implicit — each month gets a new row.
"""
import logging
from datetime import datetime, timezone

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

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