"""Background jobs — daily digest and weekly cleanup."""
import logging
from datetime import datetime, timedelta, timezone

from sqlalchemy import select, func, text

from config import settings
from db import AsyncSessionLocal, Tenant, UnansweredQuery
from image_buffer import image_buffer
from limiter import tg_rate_limiter, wa_rate_limiter
from state import telegram_apps

logger = logging.getLogger(__name__)


async def daily_digest_job():
    """Send top unanswered queries to each tenant's operator via Telegram."""
    cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
    async with AsyncSessionLocal() as db:
        result = await db.execute(
            select(Tenant).where(Tenant.active == True, Tenant.operator_chat_id != None)  # noqa: E712
        )
        tenants = result.scalars().all()

    for tenant in tenants:
        try:
            tg_app = telegram_apps.get(tenant.bot_token)
            if not tg_app:
                continue
            async with AsyncSessionLocal() as db:
                result = await db.execute(
                    select(
                        UnansweredQuery.question,
                        UnansweredQuery.intent_category,
                        func.count().label("cnt"),
                    )
                    .where(
                        UnansweredQuery.tenant_id == tenant.id,
                        UnansweredQuery.created_at >= cutoff,
                        UnansweredQuery.intent_category != "greeting",
                    )
                    .group_by(UnansweredQuery.question, UnansweredQuery.intent_category)
                    .order_by(
                        # needs_human first, then by frequency
                        text("CASE WHEN intent_category = 'needs_human' THEN 0 ELSE 1 END"),
                        text("cnt DESC"),
                    )
                    .limit(10)
                )
                rows = result.fetchall()
            if not rows:
                continue
            bot_name = tenant.name or tenant.slug
            lines = [f"📊 *Consultas sin respuesta — {bot_name}* (últimas 24h):\n"]
            intent_labels = {
                "needs_human":  "🚨 quiere humano",
                "off_topic":    "↩️ fuera de tema",
                "ambiguous":    "❓ sin info",
            }
            for i, row in enumerate(rows, 1):
                label = intent_labels.get(row.intent_category, row.intent_category)
                lines.append(f"{i}. {row.question} ({row.cnt}×) — _{label}_")
            await tg_app.bot.send_message(
                chat_id=tenant.operator_chat_id,
                text="\n".join(lines),
                parse_mode="Markdown",
            )
        except Exception:
            logger.exception("daily_digest_job failed for tenant %s", tenant.slug)


async def cleanup_job():
    """Weekly: purge old UnansweredQuery rows and sweep stale rate-limit entries."""
    cutoff = datetime.now(timezone.utc) - timedelta(days=90)
    async with AsyncSessionLocal() as db:
        await db.execute(
            text("DELETE FROM unanswered_queries WHERE created_at < :cutoff"),
            {"cutoff": cutoff},
        )
        await db.commit()
    logger.info("cleanup_job: deleted UnansweredQuery rows older than 90 days")

    tg_removed = tg_rate_limiter.sweep()
    wa_removed = wa_rate_limiter.sweep()
    buf_removed = image_buffer.sweep()
    logger.info("cleanup_job: rate_limit_sweep removed=%d TG, %d WA; image_buffer_sweep removed=%d stale entries", tg_removed, wa_removed, buf_removed)