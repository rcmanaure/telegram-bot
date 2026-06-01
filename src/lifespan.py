"""Application lifespan — startup and shutdown logic."""
import logging
from contextlib import asynccontextmanager

from apscheduler.schedulers.asyncio import AsyncIOScheduler
import sentry_sdk
from sentry_sdk.integrations.fastapi import FastApiIntegration
from sentry_sdk.integrations.sqlalchemy import SqlalchemyIntegration
from fastapi import FastAPI
from sqlalchemy import select

from config import settings
from db import init_db, Tenant
from services.jobs import daily_digest_job, cleanup_job
from services.tenant_bot import init_tenant_bot
from state import telegram_apps

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    from rag import http_client

    if settings.sentry_dsn:
        sentry_sdk.init(
            dsn=settings.sentry_dsn,
            integrations=[FastApiIntegration(), SqlalchemyIntegration()],
            traces_sample_rate=0.1,
            environment=settings.environment,
        )

    await init_db()

    # Use APP_DOMAIN directly in prod; discover via ngrok in local dev
    from services.ngrok import get_ngrok_domain
    if not settings.app_domain.startswith("localhost"):
        domain = settings.app_domain
        logger.info("Webhook domain: %s", domain)
    else:
        domain = await get_ngrok_domain(http_client) or settings.app_domain

    # Build Application per active tenant and register webhooks
    from db import AsyncSessionLocal
    async with AsyncSessionLocal() as db:
        result = await db.execute(select(Tenant).where(Tenant.active == True))  # noqa: E712
        tenants = result.scalars().all()

    for tenant in tenants:
        try:
            ok = await init_tenant_bot(tenant, domain)
            if ok:
                logger.info("Webhook set for tenant %s", tenant.slug)
        except Exception:
            logger.exception("Failed to initialize tenant %s — skipping", tenant.slug)

    # Index FAQ chunks for tenants that have example_questions
    from rag import sync_faq_chunks
    async with AsyncSessionLocal() as db:
        for tenant in tenants:
            if tenant.example_questions:
                try:
                    count = await sync_faq_chunks(db, tenant.slug, tenant.example_questions)
                    if count:
                        logger.info("Indexed %d FAQ chunks for tenant %s", count, tenant.slug)
                except Exception:
                    logger.exception("Failed to index FAQ chunks for tenant %s", tenant.slug)

    scheduler = AsyncIOScheduler(timezone="UTC", misfire_grace_time=300)
    scheduler.add_job(daily_digest_job, "cron", hour=8, minute=0, id="daily_digest")
    scheduler.add_job(cleanup_job, "cron", day_of_week="sun", hour=0, minute=0, id="weekly_cleanup")
    scheduler.start()

    logger.info("RAG Bot API ready — %d tenant(s) loaded", len(telegram_apps))
    yield

    scheduler.shutdown()

    for tg_app in telegram_apps.values():
        try:
            await tg_app.shutdown()
        except Exception:
            pass
    await http_client.aclose()