"""Webhook routes — Telegram and WhatsApp."""
import asyncio
import hmac
import logging

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from telegram import Update

from config import settings
from db import get_db, Tenant
from limiter import limiter
from state import get_app
from services.wa_processor import create_wa_adapter, handle_wa_message

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/webhook/{tenant_slug}")
@limiter.limit("20/minute")
async def telegram_webhook(
    tenant_slug: str,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    received = request.headers.get("X-Telegram-Bot-Api-Secret-Token", "")

    result = await db.execute(
        select(Tenant).where(Tenant.slug == tenant_slug, Tenant.active == True)  # noqa: E712
    )
    tenant = result.scalar_one_or_none()
    if not tenant:
        raise HTTPException(status_code=404, detail="Tenant not found")

    if not hmac.compare_digest(received, tenant.webhook_secret):
        raise HTTPException(status_code=403, detail="Invalid webhook signature")

    tg_app = get_app(tenant.bot_token)
    if not tg_app:
        logger.error("No Application for tenant %s — restart may be needed", tenant_slug)
        return {"ok": True}

    data = await request.json()
    update = Update.de_json(data, tg_app.bot)
    await tg_app.process_update(update)
    return {"ok": True}


@router.get("/webhook/{tenant_slug}/whatsapp")
async def whatsapp_webhook_verify(
    tenant_slug: str,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """WhatsApp webhook GET verification (hub.challenge)."""
    result = await db.execute(
        select(Tenant).where(Tenant.slug == tenant_slug, Tenant.active == True)  # noqa: E712
    )
    tenant = result.scalar_one_or_none()
    if not tenant:
        raise HTTPException(status_code=404, detail="Tenant not found")

    adapter = create_wa_adapter(tenant)
    if not adapter:
        raise HTTPException(status_code=404, detail="WhatsApp not configured for this tenant")

    response = await adapter.handle_verification(request)
    if response:
        return response
    raise HTTPException(status_code=403, detail="Verification failed")


@router.post("/webhook/{tenant_slug}/whatsapp")
@limiter.limit("20/minute")
async def whatsapp_webhook(
    tenant_slug: str,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """WhatsApp webhook POST — receive and process messages.

    Sync path: tenant lookup, HMAC verify, dedup check, spawn BackgroundTask.
    Returns 200 immediately so Meta doesn't retry.
    """
    # 1. Lookup tenant
    result = await db.execute(
        select(Tenant).where(Tenant.slug == tenant_slug, Tenant.active == True)  # noqa: E712
    )
    tenant = result.scalar_one_or_none()
    if not tenant:
        raise HTTPException(status_code=404, detail="Tenant not found")

    # 2. Check WA is enabled
    if "whatsapp" not in (tenant.channels or "telegram"):
        raise HTTPException(status_code=404, detail="WhatsApp not enabled for this tenant")

    # 3. Read body for HMAC verification
    body_bytes = await request.body()

    # 4. HMAC verification
    adapter = create_wa_adapter(tenant)
    if not adapter:
        raise HTTPException(status_code=404, detail="WhatsApp not configured for this tenant")

    if not adapter.verify_webhook(request, body_bytes):
        logger.warning("wa_hmac_failed tenant=%s", tenant_slug)
        raise HTTPException(status_code=403, detail="Invalid webhook signature")

    # 5. Parse payload
    try:
        data = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON payload")

    # 6. Parse incoming messages (handles status callbacks → empty list)
    try:
        messages = await adapter.parse_incoming(data)
    except ValueError:
        raise HTTPException(status_code=400, detail="Malformed webhook payload")

    # 7. Process each message in background
    for msg in messages:
        asyncio.create_task(handle_wa_message(tenant, msg))

    return {"ok": True}