"""Telegram bot initialization — shared by lifespan and admin create-tenant."""
import logging

from telegram.ext import ApplicationBuilder

from state import register_app

logger = logging.getLogger(__name__)


async def init_tenant_bot(tenant, domain: str) -> bool:
    """Build telegram Application for a tenant and register its webhook.

    Returns True on success, False on failure (logs the exception).
    """
    from bot import register_handlers

    try:
        tg_app = ApplicationBuilder().token(tenant.bot_token).build()
        register_handlers(tg_app)
        tg_app.bot_data["tenant"] = tenant
        await tg_app.initialize()
        await tg_app.bot.set_webhook(
            url=f"https://{domain}/webhook/{tenant.slug}",
            secret_token=tenant.webhook_secret,
        )
        register_app(tenant.bot_token, tg_app)
        return True
    except Exception:
        logger.exception("Failed to initialize bot for tenant %s", tenant.slug)
        return False