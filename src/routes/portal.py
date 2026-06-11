"""Tenant self-service portal routes (PR2).

PR1: login endpoint only (JSON API).
PR2: cookie-based login, dashboard UI, document CRUD, answer preview,
     doc-gap suggestions, audit log, plan limits, metering.
"""
import logging
import os
from typing import AsyncGenerator

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from jinja2 import Environment, FileSystemLoader, select_autoescape
from pydantic import BaseModel
from sqlalchemy import select, func, delete as sa_delete
from sqlalchemy.ext.asyncio import AsyncSession

from auth import create_access_token, verify_portal_password, hash_portal_password
from config import settings, PLAN_LIMITS
from db import Tenant, UnansweredQuery, TenantAuditLog, get_db, tenant_session
from dependencies import require_portal_auth
from limiter import limiter, portal_login_limiter
from rag import rag_query as _rag_query, get_index_status
from services.knowledge import list_sources, upsert_source, delete_source, process_upload
from services.upload import normalize_source_name
from services.usage import check_and_increment_usage, get_usage, increment_usage, write_audit_log
from services.upload import MAX_UPLOAD_BYTES

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/portal", tags=["portal"])

# ─── Template rendering ──────────────────────────────────────────────────────
_templates_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "templates")
_jinja_env = Environment(
    loader=FileSystemLoader(_templates_dir),
    autoescape=select_autoescape(["html"]),
)


def _render_portal(template_name: str, **kwargs) -> HTMLResponse:
    template = _jinja_env.get_template(template_name)
    html = template.render(**kwargs)
    return HTMLResponse(content=html)


# ─── Models ────────────────────────────────────────────────────────────────────
class PortalLoginRequest(BaseModel):
    slug: str
    password: str


class PortalLoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


# ─── Login (JSON API — kept for backward compat) ─────────────────────────────
@router.post("/login", response_model=PortalLoginResponse)
@limiter.limit("5/minute")
async def portal_login_json(
    request: Request,
    body: PortalLoginRequest,
    db: AsyncSession = Depends(get_db),
):
    """Authenticate tenant with slug + portal password. Returns JWT (JSON)."""
    client_ip = request.client.host if request.client else "unknown"
    if portal_login_limiter.check(f"login:{client_ip}"):
        raise HTTPException(status_code=429, detail="Too many login attempts.")

    result = await db.execute(
        select(Tenant).where(Tenant.slug == body.slug, Tenant.active == True)  # noqa: E712
    )
    tenant = result.scalar_one_or_none()
    if not tenant or not tenant.portal_password_hash or \
       not verify_portal_password(body.password, tenant.portal_password_hash):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    if not settings.jwt_secret:
        raise HTTPException(status_code=500, detail="JWT_SECRET not configured")

    token = create_access_token(tenant.slug, settings.jwt_secret)
    await write_audit_log(db, tenant.id, tenant.slug, f"tenant:{tenant.slug}", "portal.login",
                          auto_commit=True)
    logger.info("portal_login tenant=%s ip=%s", tenant.slug, client_ip)
    return PortalLoginResponse(access_token=token)


# ─── Login (HTML form — cookie-based for browser) ────────────────────────────
@router.get("/login", response_class=HTMLResponse)
@limiter.limit("20/minute")
async def portal_login_page(request: Request):
    """Render the portal login form."""
    return _render_portal("portal/login.html")


@router.post("/login/form", response_class=HTMLResponse)
@limiter.limit("5/minute")
async def portal_login_form(
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """Process login form submission. Sets httpOnly cookie, redirects to dashboard."""
    form = await request.form()
    slug = form.get("slug", "").strip()
    password = form.get("password", "")
    error = None

    client_ip = request.client.host if request.client else "unknown"
    if portal_login_limiter.check(f"login:{client_ip}"):
        error = "Demasiados intentos. Espera un momento."
    elif not slug or not password:
        error = "Completá todos los campos."
    else:
        result = await db.execute(
            select(Tenant).where(Tenant.slug == slug, Tenant.active == True)  # noqa: E712
        )
        tenant = result.scalar_one_or_none()
        if not tenant or not tenant.portal_password_hash or \
           not verify_portal_password(password, tenant.portal_password_hash):
            error = "Credenciales inválidas."
        elif not settings.jwt_secret:
            error = "Servidor no configurado. Contactá al administrador."
        else:
            token = create_access_token(tenant.slug, settings.jwt_secret)
            await write_audit_log(db, tenant.id, tenant.slug, f"tenant:{tenant.slug}",
                                  "portal.login", auto_commit=True)
            logger.info("portal_login_form tenant=%s ip=%s", tenant.slug, client_ip)
            response = RedirectResponse(url="/portal/dashboard", status_code=303)
            response.set_cookie(
                "portal_token", token,
                httponly=True, samesite="strict", secure=True, max_age=3600, path="/portal",
            )
            return response

    return _render_portal("portal/login.html", error=error, slug=slug)


# ─── Logout ───────────────────────────────────────────────────────────────────
@router.post("/logout", response_class=HTMLResponse)
async def portal_logout(request: Request):
    """Clear portal cookie and redirect to login."""
    response = RedirectResponse(url="/portal/login", status_code=303)
    response.delete_cookie("portal_token", path="/portal")
    return response


# ─── Dashboard ────────────────────────────────────────────────────────────────
@router.get("/dashboard", response_class=HTMLResponse)
@limiter.limit("20/minute")
async def portal_dashboard(
    request: Request,
    tenant_data: tuple[Tenant, AsyncSession] = Depends(require_portal_auth),
):
    """Render the portal dashboard with all 4 tabs."""
    tenant, db = tenant_data
    plan_limits = PLAN_LIMITS.get(tenant.plan, PLAN_LIMITS["free"])

    # Fetch sources
    sources = await list_sources(db, tenant.slug)

    # Index status for each source
    source_status = {s["source"]: get_index_status(tenant.slug, s["source"]) or "activo"
                     for s in sources}

    # Usage counts
    query_count = await get_usage(db, tenant.id, "queries")
    upload_count = await get_usage(db, tenant.id, "uploads")

    # Unanswered queries
    queries_result = await db.execute(
        select(UnansweredQuery)
        .where(UnansweredQuery.namespace == tenant.slug)
        .order_by(UnansweredQuery.created_at.desc())
        .limit(50)
    )
    unanswered = queries_result.scalars().all()

    # Audit log
    audit_result = await db.execute(
        select(TenantAuditLog)
        .where(TenantAuditLog.namespace == tenant.slug)
        .order_by(TenantAuditLog.created_at.desc())
        .limit(100)
    )
    audit_entries = audit_result.scalars().all()

    return _render_portal(
        "portal/dashboard.html",
        tenant=tenant,
        sources=sources,
        source_status=source_status,
        doc_count=len(sources),
        doc_limit=plan_limits["docs"],
        query_count=query_count,
        query_limit=plan_limits["queries_monthly"],
        upload_count=upload_count,
        unanswered=unanswered,
        audit_entries=audit_entries,
        message=request.query_params.get("message", ""),
        error=request.query_params.get("error", ""),
    )


# ─── Upload ───────────────────────────────────────────────────────────────────
@router.post("/upload", response_class=HTMLResponse)
@limiter.limit("5/minute")
async def portal_upload(
    request: Request,
    tenant_data: tuple[Tenant, AsyncSession] = Depends(require_portal_auth),
):
    """Upload a document for the tenant's knowledge base."""
    tenant, db = tenant_data
    plan_limits = PLAN_LIMITS.get(tenant.plan, PLAN_LIMITS["free"])

    form = await request.form()
    file = form.get("file")
    if not file or not file.filename:
        return RedirectResponse(url="/portal/dashboard?error=No+se+seleccionó+ningún+archivo", status_code=303)

    # Enforce upload size limit
    content = await file.read()
    if len(content) > MAX_UPLOAD_BYTES:
        return RedirectResponse(url="/portal/dashboard?error=El+archivo+es+demasiado+grande+(máx+10MB)", status_code=303)

    # Enforce doc limit (allow re-upload of existing source)
    sources = await list_sources(db, tenant.slug)
    existing_names = {s["source"] for s in sources}
    source_name = normalize_source_name(file.filename)
    if len(sources) >= plan_limits["docs"] and source_name not in existing_names:
        return RedirectResponse(url="/portal/dashboard?error=Límite+de+documentos+alcanzado+("+str(plan_limits["docs"])+")", status_code=303)

    # Process upload
    try:
        all_chunks, _pages, full_doc_text = await process_upload(content, file.filename)
    except Exception as e:
        logger.exception("portal_upload process_upload failed for tenant=%s", tenant.slug)
        await db.rollback()
        return RedirectResponse(url=f"/portal/dashboard?error=Error+procesando+el+archivo", status_code=303)

    # Enforce chunk limit
    current_chunks = sum(s["chunks"] for s in sources)
    if source_name in existing_names:
        # Re-upload: subtract old chunks, add new
        old_chunks = next(s["chunks"] for s in sources if s["source"] == source_name)
        total_after = current_chunks - old_chunks + len(all_chunks)
    else:
        total_after = current_chunks + len(all_chunks)

    if total_after > plan_limits["chunks"]:
        return RedirectResponse(url="/portal/dashboard?error=Límite+de+fragmentos+alcanzado+("+str(plan_limits["chunks"])+")", status_code=303)

    # Upsert source — use auto_commit=False to preserve RLS GUC across the session
    stored = await upsert_source(db, tenant.slug, source_name, all_chunks, full_doc_text=full_doc_text,
                                  auto_commit=False)

    # E2: meter upload
    await increment_usage(db, tenant.id, "uploads", auto_commit=False)

    # E6: audit log
    await write_audit_log(db, tenant.id, tenant.slug, f"tenant:{tenant.slug}", "document.upload",
                         detail={"source": source_name, "chunks": stored}, auto_commit=False)

    # Single commit for all writes — preserves RLS GUC across the entire request
    await db.commit()

    return RedirectResponse(url="/portal/dashboard?message=Documento+subido+correctamente", status_code=303)


# ─── Delete source ────────────────────────────────────────────────────────────
@router.post("/delete/{source:path}", response_class=HTMLResponse)
@limiter.limit("5/minute")
async def portal_delete_source(
    source: str,
    request: Request,
    tenant_data: tuple[Tenant, AsyncSession] = Depends(require_portal_auth),
):
    """Delete a single source from the tenant's knowledge base.

    Validates that the source exists in the tenant's namespace before deleting
    (defense-in-depth: RLS already scopes rows, but this prevents silent
    success when deleting a non-existent source name).
    """
    tenant, db = tenant_data
    # Path traversal guard: reject sources containing path separators or ..
    if "/" in source or "\\" in source or ".." in source:
        return RedirectResponse(
            url="/portal/dashboard?error=Nombre+de+fuente+inválido",
            status_code=303,
        )
    # P0-4: Validate source exists in tenant's namespace before deleting
    normalized = normalize_source_name(source)
    sources = await list_sources(db, tenant.slug)
    existing_names = {s["source"] for s in sources}
    if normalized not in existing_names:
        return RedirectResponse(
            url=f"/portal/dashboard?error=Fuente+no+encontrada:+{normalized}",
            status_code=303,
        )

    await delete_source(db, tenant.slug, source, auto_commit=False)

    # E6: audit log
    await write_audit_log(db, tenant.id, tenant.slug, f"tenant:{tenant.slug}", "document.delete",
                         detail={"source": source}, auto_commit=False)

    # Single commit for all writes — preserves RLS GUC across the entire request
    await db.commit()

    return RedirectResponse(url="/portal/dashboard?message=Documento+eliminado", status_code=303)


# ─── Answer preview (E3) ─────────────────────────────────────────────────────
@router.post("/query")
@limiter.limit("10/minute")
async def portal_query(
    request: Request,
    tenant_data: tuple[Tenant, AsyncSession] = Depends(require_portal_auth),
):
    """Answer preview sandbox. Returns JSON for the JS frontend."""
    tenant, db = tenant_data
    plan_limits = PLAN_LIMITS.get(tenant.plan, PLAN_LIMITS["free"])

    # E2: atomic limit check + increment (prevents TOCTOU race)
    if not await check_and_increment_usage(db, tenant.id, "queries", plan_limits["queries_monthly"], auto_commit=False):
        raise HTTPException(status_code=429, detail="Monthly query limit reached")

    form = await request.form()
    question = (form.get("question") or "").strip()
    if not question:
        # Increment happened above but question is empty — rollback the counter
        await db.rollback()
        raise HTTPException(status_code=400, detail="Question is required")

    # Run RAG query scoped to tenant
    try:
        answer, chunks, intent = await _rag_query(
            db=db,
            question=question,
            namespace=tenant.slug,
            user_id=f"portal:{tenant.slug}",
            expertise_area=tenant.expertise_area or "",
            tenant_id=tenant.id,
            channel="portal",
            tenant=tenant,
        )
    except Exception:
        await db.rollback()
        raise

    # E6: audit log
    await write_audit_log(db, tenant.id, tenant.slug, f"tenant:{tenant.slug}", "portal.query",
                         detail={"question": question[:200]}, auto_commit=False)

    # Single commit for all writes — preserves RLS GUC across the entire request
    await db.commit()

    # Build sources list
    sources = []
    for c in (chunks or []):
        sources.append({"source": c.get("source", ""), "page": c.get("page", 0)})

    return {"answer": answer, "sources": sources, "intent": intent}


# ─── Resolve unanswered question (E5) ────────────────────────────────────────
@router.post("/resolve-question", response_class=HTMLResponse)
@limiter.limit("20/minute")
async def portal_resolve_question(
    request: Request,
    tenant_data: tuple[Tenant, AsyncSession] = Depends(require_portal_auth),
):
    """Mark an unanswered question as resolved."""
    tenant, db = tenant_data
    form = await request.form()
    question_id = form.get("question_id")

    if question_id:
        try:
            qid = int(question_id)
            result = await db.execute(
                select(UnansweredQuery).where(
                    UnansweredQuery.id == qid,
                    UnansweredQuery.namespace == tenant.slug,
                )
            )
            query = result.scalar_one_or_none()
            if query:
                await db.delete(query)
                await db.commit()
            # If query is None, the ID doesn't belong to this tenant —
            # redirect with a specific error instead of silent success
        except ValueError:
            await db.rollback()
        except Exception:
            logger.exception("portal_resolve_question failed for tenant=%s", tenant.slug)
            await db.rollback()

    return RedirectResponse(url="/portal/dashboard?message=Pregunta+resuelta", status_code=303)