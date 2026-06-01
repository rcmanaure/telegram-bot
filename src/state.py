"""
Shared in-process state — telegram app registry.

Single-worker only (1 uvicorn process). telegram_apps is a module-level
dict keyed by bot_token. Accessor functions keep the import surface clean.
"""
import logging

logger = logging.getLogger(__name__)

# bot_token → telegram.ext.Application
telegram_apps: dict[str, object] = {}


def register_app(bot_token: str, app: object) -> None:
    telegram_apps[bot_token] = app
    logger.debug("register_app token=%s…", bot_token[:8])


def get_app(bot_token: str) -> object | None:
    return telegram_apps.get(bot_token)


def update_tenant_cache(tenant) -> None:
    """Refresh the cached Tenant object in an already-registered bot."""
    app = telegram_apps.get(tenant.bot_token)
    if app is not None:
        app.bot_data["tenant"] = tenant


def remove_app(bot_token: str) -> None:
    telegram_apps.pop(bot_token, None)