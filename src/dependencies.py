"""FastAPI dependencies — shared across route modules."""
import hashlib
from typing import AsyncGenerator

import jwt as pyjwt
from fastapi import Depends, Header, HTTPException, Request
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from auth import decode_access_token
from config import settings
from db import Tenant, get_db, tenant_session


async def require_tenant(
    x_api_key: str = Header(..., alias="X-API-Key"),
    db: AsyncSession = Depends(get_db),
) -> Tenant:
    """Resolve X-API-Key header to an active Tenant row."""
    key_hash = hashlib.sha256(x_api_key.encode()).hexdigest()  # noqa: S324
    result = await db.execute(
        select(Tenant).where(Tenant.api_key_hash == key_hash, Tenant.active == True)  # noqa: E712
    )
    tenant = result.scalar_one_or_none()
    if not tenant:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return tenant


async def require_tenant_session(
    authorization: str = Header(..., alias="Authorization"),
    db: AsyncSession = Depends(get_db),  # admin session for tenant lookup
) -> AsyncGenerator[tuple[Tenant, AsyncSession], None]:
    """JWT-based portal auth (API consumers). Returns (tenant, tenant_scoped_session).

    Namespace is ALWAYS derived from the JWT subject, never from request body.
    Re-checks tenant.active on every request (revocation support).
    Opens a tenant_session(slug) with RLS GUC set.
    """
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing Bearer token")

    token = authorization[7:]
    try:
        payload = decode_access_token(token, settings.jwt_secret)
    except pyjwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    slug = payload.get("sub")
    if not slug:
        raise HTTPException(status_code=401, detail="Invalid token: missing subject")

    # Re-check tenant is active (supports revocation without waiting for TTL)
    result = await db.execute(
        select(Tenant).where(Tenant.slug == slug, Tenant.active == True)  # noqa: E712
    )
    tenant = result.scalar_one_or_none()
    if not tenant:
        raise HTTPException(status_code=401, detail="Tenant inactive or not found")

    # Open tenant-scoped session with RLS GUC
    async with tenant_session(slug) as tenant_db:
        yield tenant, tenant_db


async def require_portal_auth(
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> AsyncGenerator[tuple[Tenant, AsyncSession], None]:
    """Portal browser auth. Reads JWT from Authorization header or portal_token cookie.

    On auth failure, redirects to /portal/login (303) instead of returning JSON 401.
    Used by all portal HTML routes. API consumers should use require_tenant_session.
    """
    token = None
    auth_header = request.headers.get("authorization", "")
    if auth_header.startswith("Bearer "):
        token = auth_header[7:]
    elif "portal_token" in request.cookies:
        token = request.cookies["portal_token"]

    if not token:
        raise HTTPException(status_code=303, detail="Login required",
                            headers={"Location": "/portal/login"})

    try:
        payload = decode_access_token(token, settings.jwt_secret)
    except pyjwt.InvalidTokenError:
        raise HTTPException(status_code=303, detail="Invalid or expired token",
                            headers={"Location": "/portal/login"})

    slug = payload.get("sub")
    if not slug:
        raise HTTPException(status_code=303, detail="Invalid token",
                            headers={"Location": "/portal/login"})

    # Re-check tenant is active
    result = await db.execute(
        select(Tenant).where(Tenant.slug == slug, Tenant.active == True)  # noqa: E712
    )
    tenant = result.scalar_one_or_none()
    if not tenant:
        raise HTTPException(status_code=303, detail="Tenant inactive or not found",
                            headers={"Location": "/portal/login"})

    # Open tenant-scoped session with RLS GUC
    async with tenant_session(slug) as tenant_db:
        yield tenant, tenant_db