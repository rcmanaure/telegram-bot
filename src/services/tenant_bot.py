"""Telegram bot initialization — shared by lifespan and admin create-tenant."""
import logging

from telegram.ext import ApplicationBuilder

from state import register_app

logger = logging.getLogger(__name__)


async def init_tenant_bot(tenant, domain: str) -> bool:
    """Build telegram Application for a tenant and register its webhook.

    Returns True on success, False on failure (logs the exception).
    If an Application for this bot_token already exists, shuts it down first.
    """
    from bot import register_handlers
    from state import get_app

    # Shut down existing Application if one exists for this bot_token
    existing_app = get_app(tenant.bot_token)
    if existing_app is not None:
        try:
            await existing_app.shutdown()
        except Exception as e:
            logger.warning("Failed to shut down existing app for tenant %s: %s", tenant.slug, e)

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