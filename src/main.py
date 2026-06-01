"""FastAPI backend — multi-tenant RAG API. Slim wiring module."""
import os
import logging

from logging_config import setup_logging
setup_logging()

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from slowapi.errors import RateLimitExceeded

from limiter import limiter, rate_limit_handler
from lifespan import lifespan
from routes import api, admin, webhook

logger = logging.getLogger(__name__)

# ─── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="RAG Bot API",
    description="Multi-tenant RAG bot platform",
    version="2.0.0",
    lifespan=lifespan,
)
app.state.limiter = limiter

# ─── Static files ─────────────────────────────────────────────────────────────

_static_dir = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=_static_dir), name="static")

# ─── Routers ──────────────────────────────────────────────────────────────────

app.include_router(api.router)
app.include_router(admin.router)
app.include_router(webhook.router)

# ─── Exception handlers ──────────────────────────────────────────────────────

@app.exception_handler(RateLimitExceeded)
async def _rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return rate_limit_handler(request, exc)