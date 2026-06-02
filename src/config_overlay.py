"""
Runtime config overlay: DB-stored settings override .env values without restart.

Resolution order:
  1. _overlay dict (loaded from system_config table at startup + after each save)
  2. fallback argument (from settings object, i.e., .env)

Usage:
    from config_overlay import get_setting
    api_key = get_setting("llm_api_key", settings.effective_llm_api_key)
"""

import logging
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)

# Mutable in-process overlay. Single-worker guarantee: no race conditions.
_overlay: dict[str, str] = {}


def get_setting(key: str, fallback: str = "") -> str:
    """Return DB override if present, else fallback (.env value).

    Note: `or` semantics mean empty-string overrides fall through to fallback.
    This is intentional — the admin UI skips empty form fields to preserve
    existing values, and intentional "unset" requires deleting the DB row.
    """
    return _overlay.get(key) or fallback


def get_setting_int(key: str, fallback: int) -> int:
    """Return DB override cast to int if present, else fallback."""
    val = _overlay.get(key)
    if val is not None:
        try:
            return int(val)
        except ValueError:
            logger.warning("config_overlay: key=%s value=%r is not an int, using fallback", key, val)
    return fallback


async def reload_from_db(db: AsyncSession) -> None:
    """Load all SystemConfig rows into the in-process overlay.
    Per-row decrypt errors are caught and logged — one bad row does not abort load.
    """
    from sqlalchemy import select, text
    from crypto import decrypt_value

    try:
        result = await db.execute(
            text("SELECT key, encrypted_value FROM system_config")
        )
        rows = result.fetchall()
    except Exception as e:
        logger.warning("config_overlay: reload_from_db failed to read table: %s", e)
        return

    new_overlay: dict[str, str] = {}
    for row in rows:
        try:
            new_overlay[row.key] = decrypt_value(row.encrypted_value)
        except Exception as e:
            logger.warning("config_overlay: failed to decrypt key=%s: %s — skipping", row.key, e)

    _overlay.clear()
    _overlay.update(new_overlay)
    logger.info("config_overlay: loaded %d settings from DB", len(_overlay))
