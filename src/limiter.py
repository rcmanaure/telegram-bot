"""Unified rate limiting — Telegram and WhatsApp.

Single RateLimiter class used by both channels.
Uses time.monotonic() for clock-immune tracking, >= MAX to block
(fixes off-by-one where TG allowed 21 messages).
"""
import logging
import time
from collections import defaultdict, deque

from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

logger = logging.getLogger(__name__)

MAX_MESSAGES_PER_WINDOW = 20
RATE_LIMIT_WINDOW_SECONDS = 60

# ─── Per-user rate limiter (in-memory, single-worker) ──────────────────────────

class RateLimiter:
    """Generic sliding-window rate limiter.

    check(key) returns True if the user is rate-limited (over limit).
    sweep() removes stale entries — called by the weekly cleanup job.
    """

    def __init__(self, max_messages: int = MAX_MESSAGES_PER_WINDOW,
                 window_seconds: int = RATE_LIMIT_WINDOW_SECONDS):
        self._max = max_messages
        self._window = window_seconds
        self._timestamps: dict[str, deque] = defaultdict(deque)

    def check(self, key: str) -> bool:
        """Return True if key is rate-limited (>= max_messages in window)."""
        now = time.monotonic()
        timestamps = self._timestamps[key]
        timestamps.append(now)
        cutoff = now - self._window
        while timestamps and timestamps[0] <= cutoff:
            timestamps.popleft()
        return len(timestamps) >= self._max

    def sweep(self) -> int:
        """Remove entries whose window has fully expired. Returns count removed."""
        now = time.monotonic()
        cutoff = now - self._window
        stale = [k for k, v in self._timestamps.items() if not v or v[-1] <= cutoff]
        for k in stale:
            del self._timestamps[k]
        if stale:
            logger.debug("rate_limit_sweep removed=%d remaining=%d", len(stale), len(self._timestamps))
        return len(stale)


# ─── SlowAPI limiter for HTTP endpoints ────────────────────────────────────────

limiter = Limiter(key_func=get_remote_address)


def rate_limit_handler(request, exc: RateLimitExceeded):
    """FastAPI exception handler for slowapi rate limits."""
    from fastapi.responses import JSONResponse
    return JSONResponse(status_code=429, content={"detail": "Rate limit exceeded"})


# ─── Module-level instances ──────────────────────────────────────────────────────

# Telegram rate limiter
tg_rate_limiter = RateLimiter()

# WhatsApp rate limiter (separate keys: namespace:user_id)
wa_rate_limiter = RateLimiter()