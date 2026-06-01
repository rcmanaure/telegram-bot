"""Ngrok public URL discovery — shared by lifespan and admin."""
import asyncio
import logging

from config import settings

logger = logging.getLogger(__name__)


async def get_ngrok_domain(client, max_retries: int = 20) -> str | None:
    """Query ngrok's local API (http://ngrok:4040) to get the public HTTPS URL.

    Retries for up to max_retries seconds to handle startup ordering.
    Returns None if ngrok is unavailable.
    """
    for _ in range(max_retries):
        try:
            resp = await client.get("http://ngrok:4040/api/tunnels", timeout=2.0)
            for tunnel in resp.json().get("tunnels", []):
                if tunnel.get("proto") == "https":
                    url = tunnel["public_url"].replace("https://", "")
                    logger.info("ngrok public domain: %s", url)
                    return url
        except Exception:
            pass
        await asyncio.sleep(1)
    logger.warning("ngrok not available — falling back to APP_DOMAIN=%s", settings.app_domain)
    return None