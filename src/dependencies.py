"""FastAPI dependencies — shared across route modules."""
import hashlib
from typing import AsyncGenerator

import jwt as pyjwt
from fastapi import Depends, Header, HTTPException
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
    """JWT-based portal auth. Returns (tenant, tenant_scoped_session).

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