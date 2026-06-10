"""Tenant self-service portal routes (PR1).

Minimal PR1: login endpoint only. Portal UI and full CRUD come in PR2.
"""
import logging

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from auth import create_access_token, verify_portal_password
from config import settings
from db import Tenant, get_db
from limiter import limiter, portal_login_limiter

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/portal", tags=["portal"])


class PortalLoginRequest(BaseModel):
    slug: str
    password: str


class PortalLoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


@router.post("/login")
@limiter.limit("5/minute")
async def portal_login(
    request: Request,
    body: PortalLoginRequest,
    db: AsyncSession = Depends(get_db),
):
    """Authenticate tenant with slug + portal password. Returns JWT."""
    # Per-IP rate limit (second layer after SlowAPI)
    client_ip = request.client.host if request.client else "unknown"
    if portal_login_limiter.check(f"login:{client_ip}"):
        raise HTTPException(status_code=429, detail="Too many login attempts. Try again later.")

    # Look up tenant
    result = await db.execute(
        select(Tenant).where(Tenant.slug == body.slug, Tenant.active == True)  # noqa: E712
    )
    tenant = result.scalar_one_or_none()
    if not tenant:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    # Verify portal password
    if not tenant.portal_password_hash:
        raise HTTPException(status_code=401, detail="Portal access not configured for this tenant")

    if not verify_portal_password(body.password, tenant.portal_password_hash):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    # Issue JWT
    if not settings.jwt_secret:
        raise HTTPException(status_code=500, detail="JWT_SECRET not configured")

    token = create_access_token(tenant.slug, settings.jwt_secret)
    logger.info("portal_login tenant=%s ip=%s", tenant.slug, client_ip)
    return PortalLoginResponse(access_token=token)