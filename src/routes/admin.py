"""Admin UI routes — tenant management, document upload, queries."""
import hashlib
import logging
import os
import secrets as secrets_mod

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from jinja2 import Environment, FileSystemLoader, select_autoescape
from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from config import settings
from db import get_db, AsyncSessionLocal, DocumentChunk, Tenant, UnansweredQuery
from limiter import limiter
from rag import chunk_text, index_chunks, sync_faq_chunks
from services.ngrok import get_ngrok_domain
from services.tenant_bot import init_tenant_bot
from services.upload import MAX_UPLOAD_BYTES, _IMAGE_EXTS, describe_image_for_upload, process_uploaded_file
from state import get_app

logger = logging.getLogger(__name__)
router = APIRouter()

VALID_PLANS = ("free", "basic", "pro")
_basic = HTTPBasic()

# ─── Jinja2 templates ────────────────────────────────────────────────────────────

_templates_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "templates")
_jinja_env = Environment(
    loader=FileSystemLoader(_templates_dir),
    autoescape=select_autoescape(["html"]),
)


def _render_admin(tenants, doc_stats=None, message="", error="", new_api_key=""):
    """Render admin panel HTML via Jinja2 template. All values auto-escaped."""
    template = _jinja_env.get_template("admin/index.html")
    html = template.render(
        tenants=tenants,
        doc_stats=doc_stats or {},
        message=message,
        error=error,
        new_api_key=new_api_key,
    )
    return HTMLResponse(content=html)


def _render_queries(tenant, queries):
    """Render queries page via Jinja2 template."""
    template = _jinja_env.get_template("admin/queries.html")
    html = template.render(tenant_slug=tenant.slug, queries=queries)
    return HTMLResponse(content=html)


# ─── Admin auth ────────────────────────────────────────────────────────────────

def _require_admin(credentials: HTTPBasicCredentials = Depends(_basic)):
    ok = secrets_mod.compare_digest(
        credentials.password.encode(), settings.admin_password.encode()
    )
    if credentials.username != "admin" or not ok:
        raise HTTPException(
            status_code=401,
            detail="Acceso denegado",
            headers={"WWW-Authenticate": "Basic"},
        )


# ─── Admin helpers ─────────────────────────────────────────────────────────────

async def _all_tenants(db: AsyncSession) -> list:
    result = await db.execute(select(Tenant).order_by(Tenant.id))
    return result.scalars().all()


async def _get_all_doc_stats(db: AsyncSession, tenants=None) -> dict[int, list[dict]]:
    """Return {tenant_id: [{source, chunks}, ...]} from document_chunks.

    If tenants list is provided, skips the redundant _all_tenants query.
    """
    result = await db.execute(
        select(
            DocumentChunk.namespace,
            DocumentChunk.source,
            func.count(DocumentChunk.id).label("chunks"),
        )
        .group_by(DocumentChunk.namespace, DocumentChunk.source)
        .order_by(DocumentChunk.namespace)
    )
    rows = result.fetchall()
    if tenants is None:
        tenants = await _all_tenants(db)
    slug_to_id = {t.slug: t.id for t in tenants}
    stats: dict[int, list[dict]] = {}
    for r in rows:
        tid = slug_to_id.get(r.namespace)
        if tid is not None:
            stats.setdefault(tid, []).append({"source": r.source, "chunks": r.chunks})
    return stats


async def _admin_context(db: AsyncSession) -> tuple:
    """Fetch (tenants, doc_stats) in one pass — eliminates redundant query."""
    tenants = await _all_tenants(db)
    doc_stats = await _get_all_doc_stats(db, tenants=tenants)
    return tenants, doc_stats


# ─── Admin routes ──────────────────────────────────────────────────────────────

@router.get("/admin", response_class=HTMLResponse, include_in_schema=False)
async def admin_panel(
    db: AsyncSession = Depends(get_db),
    _: None = Depends(_require_admin),
):
    tenants, doc_stats = await _admin_context(db)
    return _render_admin(tenants, doc_stats=doc_stats)


@router.post("/admin/tenants", response_class=HTMLResponse, include_in_schema=False)
async def admin_create_tenant(
    request: Request,
    db: AsyncSession = Depends(get_db),
    _: None = Depends(_require_admin),
):
    form = await request.form()
    slug = (form.get("slug") or "").strip()
    bot_token = (form.get("bot_token") or "").strip()
    expertise_area = (form.get("expertise_area") or "").strip()
    plan = (form.get("plan") or "free").strip()
    contact_url = (form.get("contact_url") or "").strip() or None
    operator_chat_id = (form.get("operator_chat_id") or "").strip() or None
    raw_questions = (form.get("example_questions") or "").strip()
    example_questions = [q.strip() for q in raw_questions.splitlines() if q.strip()][:5] or None

    if not slug or not bot_token:
        tenants, doc_stats = await _admin_context(db)
        return _render_admin(tenants, doc_stats=doc_stats, error="Slug y Bot Token son obligatorios.")

    if plan not in VALID_PLANS:
        tenants, doc_stats = await _admin_context(db)
        return _render_admin(tenants, doc_stats=doc_stats, error=f"Plan inválido '{plan}'. Opciones: {', '.join(VALID_PLANS)}.")

    if contact_url and not contact_url.startswith(("http://", "https://")):
        tenants, doc_stats = await _admin_context(db)
        return _render_admin(tenants, doc_stats=doc_stats, error="URL de contacto debe empezar con http:// o https://")

    existing = await db.execute(select(Tenant).where(Tenant.slug == slug))
    if existing.scalar_one_or_none():
        tenants, doc_stats = await _admin_context(db)
        return _render_admin(tenants, doc_stats=doc_stats, error=f"Ya existe un tenant con slug '{slug}'.")

    raw_api_key = secrets_mod.token_urlsafe(32)
    api_key_hash = hashlib.sha256(raw_api_key.encode()).hexdigest()
    webhook_secret = secrets_mod.token_urlsafe(32)

    tenant = Tenant(
        slug=slug,
        api_key_hash=api_key_hash,
        webhook_secret=webhook_secret,
        bot_token=bot_token,
        expertise_area=expertise_area,
        contact_url=contact_url,
        example_questions=example_questions,
        operator_chat_id=operator_chat_id,
        plan=plan,
        active=True,
    )
    db.add(tenant)
    await db.commit()
    await db.refresh(tenant)

    if example_questions:
        await sync_faq_chunks(db, tenant.slug, example_questions)

    # Discover current public domain from ngrok or settings
    from rag import http_client
    domain = await get_ngrok_domain(http_client, max_retries=1) or settings.app_domain

    ok = await init_tenant_bot(tenant, domain)
    if not ok:
        tenants, doc_stats = await _admin_context(db)
        return _render_admin(
            tenants, doc_stats=doc_stats,
            error=f"Tenant '{slug}' creado en DB pero falló al inicializar el bot. Revisá el Bot Token.",
            new_api_key=raw_api_key,
        )

    logger.info("Admin created tenant %s", slug)
    tenants, doc_stats = await _admin_context(db)
    return _render_admin(
        tenants, doc_stats=doc_stats,
        message=f"✓ Tenant '{slug}' creado y bot registrado.",
        new_api_key=raw_api_key,
    )


@router.post("/admin/tenant/{tenant_id}", response_class=HTMLResponse, include_in_schema=False)
async def admin_update_tenant(
    tenant_id: int,
    request: Request,
    db: AsyncSession = Depends(get_db),
    _: None = Depends(_require_admin),
):
    form = await request.form()
    expertise_area = form.get("expertise_area", "")
    contact_url = (form.get("contact_url") or "").strip() or None
    operator_chat_id = (form.get("operator_chat_id") or "").strip() or None
    raw_questions = (form.get("example_questions") or "").strip()
    example_questions = [q.strip() for q in raw_questions.splitlines() if q.strip()][:5] or None

    # WhatsApp / multi-channel fields
    wa_phone_number_id = (form.get("wa_phone_number_id") or "").strip() or None
    wa_access_token = (form.get("wa_access_token") or "").strip() or None
    wa_app_secret = (form.get("wa_app_secret") or "").strip() or None
    wa_verify_token = (form.get("wa_verify_token") or "").strip() or None
    wa_reengagement_template = (form.get("wa_reengagement_template") or "").strip() or None

    # Vision / web search
    web_search_enabled = form.get("web_search_enabled") == "on"

    if contact_url and not contact_url.startswith(("http://", "https://")):
        tenants, doc_stats = await _admin_context(db)
        return _render_admin(tenants, doc_stats=doc_stats, error="URL de contacto debe empezar con http:// o https://")

    result = await db.execute(select(Tenant).where(Tenant.id == tenant_id))
    tenant = result.scalar_one_or_none()
    if not tenant:
        raise HTTPException(404, "Tenant not found")

    tenant.expertise_area = expertise_area
    tenant.contact_url = contact_url
    tenant.operator_chat_id = operator_chat_id
    tenant.example_questions = example_questions
    tenant.wa_phone_number_id = wa_phone_number_id
    tenant.wa_access_token = wa_access_token
    tenant.wa_app_secret = wa_app_secret
    tenant.wa_verify_token = wa_verify_token
    tenant.wa_reengagement_template = wa_reengagement_template
    tenant.web_search_enabled = web_search_enabled

    # Enable WhatsApp channel if WA credentials are provided
    if wa_phone_number_id and wa_access_token:
        tenant.channels = "telegram,whatsapp"
    else:
        tenant.channels = "telegram"
    db.add(tenant)
    await db.commit()
    await db.refresh(tenant)

    await sync_faq_chunks(db, tenant.slug, example_questions)

    tg_app = get_app(tenant.bot_token)
    if tg_app:
        tg_app.bot_data["tenant"] = tenant

    tenants, doc_stats = await _admin_context(db)
    return _render_admin(tenants, doc_stats=doc_stats, message=f"✓ Tenant '{tenant.slug}' actualizado.")


@router.get("/admin/queries/{tenant_id}", response_class=HTMLResponse, include_in_schema=False)
async def admin_queries(
    tenant_id: int,
    db: AsyncSession = Depends(get_db),
    _: None = Depends(_require_admin),
):
    result = await db.execute(select(Tenant).where(Tenant.id == tenant_id))
    tenant = result.scalar_one_or_none()
    if not tenant:
        raise HTTPException(404, "Tenant not found")

    result = await db.execute(
        select(UnansweredQuery)
        .where(UnansweredQuery.tenant_id == tenant_id)
        .order_by(UnansweredQuery.created_at.desc())
        .limit(100)
    )
    queries = result.scalars().all()
    return _render_queries(tenant, queries)


@router.post("/admin/upload/{tenant_id}", response_class=HTMLResponse, include_in_schema=False)
async def admin_upload_document(
    tenant_id: int,
    request: Request,
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
    _: None = Depends(_require_admin),
):
    result = await db.execute(select(Tenant).where(Tenant.id == tenant_id))
    tenant = result.scalar_one_or_none()
    if not tenant:
        tenants, doc_stats = await _admin_context(db)
        return _render_admin(tenants, doc_stats=doc_stats, error=f"Tenant {tenant_id} no encontrado.")

    content = await file.read()
    if len(content) > MAX_UPLOAD_BYTES:
        tenants, doc_stats = await _admin_context(db)
        return _render_admin(tenants, doc_stats=doc_stats, error="Archivo demasiado grande. Máximo 10MB.")

    try:
        fname_lower = file.filename.lower()
        if any(fname_lower.endswith(ext) for ext in _IMAGE_EXTS):
            description = await describe_image_for_upload(content, fname_lower)
            all_chunks = chunk_text(description, source=file.filename, page=1)
            pages_processed = 1
        else:
            all_chunks, pages_processed = process_uploaded_file(content, fname_lower, file.filename)
    except HTTPException as e:
        tenants, doc_stats = await _admin_context(db)
        return _render_admin(tenants, doc_stats=doc_stats, error=e.detail)

    if not all_chunks:
        tenants, doc_stats = await _admin_context(db)
        return _render_admin(tenants, doc_stats=doc_stats, error="No se pudo extraer texto del archivo.")

    # Upsert: delete old chunks for this source, then insert new ones atomically
    await db.execute(
        text("DELETE FROM document_chunks WHERE namespace = :ns AND source = :src"),
        {"ns": tenant.slug, "src": file.filename},
    )
    stored = await index_chunks(db, all_chunks, tenant.slug, auto_commit=False)
    await db.commit()
    logger.info("Admin uploaded %s for tenant %s (%d chunks)", file.filename, tenant.slug, stored)

    tenants, doc_stats = await _admin_context(db)
    return _render_admin(
        tenants, doc_stats=doc_stats,
        message=f"✓ {stored} chunks indexados desde {file.filename} para '{tenant.slug}'.",
    )


@router.post("/admin/delete-docs/{tenant_id}", response_class=HTMLResponse, include_in_schema=False)
async def admin_delete_docs(
    tenant_id: int,
    db: AsyncSession = Depends(get_db),
    _: None = Depends(_require_admin),
):
    result = await db.execute(select(Tenant).where(Tenant.id == tenant_id))
    tenant = result.scalar_one_or_none()
    if not tenant:
        raise HTTPException(404, "Tenant not found")

    ns = tenant.slug
    await db.execute(text("DELETE FROM document_chunks WHERE namespace = :ns"), {"ns": ns})
    await db.execute(text("DELETE FROM conversations WHERE namespace = :ns"), {"ns": ns})
    await db.commit()

    logger.info("Admin deleted all docs for tenant %s", tenant.slug)
    tenants, doc_stats = await _admin_context(db)
    return _render_admin(
        tenants, doc_stats=doc_stats,
        message=f"✓ Documentos y conversaciones eliminados para '{tenant.slug}'.",
    )


@router.get("/admin/template", include_in_schema=False)
async def admin_download_template(_: None = Depends(_require_admin)):
    return FileResponse("documents/plantilla.md", filename="plantilla.md", media_type="text/markdown")