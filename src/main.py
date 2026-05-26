"""
FastAPI backend — multi-tenant RAG API.
"""
import asyncio
import hashlib
import hmac
import io
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone

from logging_config import setup_logging
setup_logging()

from apscheduler.schedulers.asyncio import AsyncIOScheduler
import sentry_sdk
from sentry_sdk.integrations.fastapi import FastApiIntegration
from sentry_sdk.integrations.sqlalchemy import SqlalchemyIntegration
import secrets as secrets_mod
from fastapi import FastAPI, Header, HTTPException, UploadFile, File, Form, Depends, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel
from pypdf import PdfReader
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from sqlalchemy import select, func, text
from sqlalchemy.ext.asyncio import AsyncSession
from telegram import Update
from telegram.ext import ApplicationBuilder

from config import settings
from db import init_db, get_db, AsyncSessionLocal, DocumentChunk, Tenant, UnansweredQuery
from rag import chunk_text, index_chunks

logger = logging.getLogger(__name__)

MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB

# ─── Rate limiter ─────────────────────────────────────────────────────────────

limiter = Limiter(key_func=get_remote_address)

# ─── Per-tenant Application registry ─────────────────────────────────────────

telegram_apps: dict[str, object] = {}  # bot_token → Application

# ─── Auth dependency ──────────────────────────────────────────────────────────

async def require_tenant(
    x_api_key: str = Header(..., alias="X-API-Key"),
    db: AsyncSession = Depends(get_db),
) -> Tenant:
    key_hash = hashlib.sha256(x_api_key.encode()).hexdigest()  # noqa: S324
    result = await db.execute(
        select(Tenant).where(Tenant.api_key_hash == key_hash, Tenant.active == True)
    )
    tenant = result.scalar_one_or_none()
    if not tenant:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return tenant


# ─── Background jobs ─────────────────────────────────────────────────────────

async def daily_digest_job():
    cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
    async with AsyncSessionLocal() as db:
        result = await db.execute(
            select(Tenant).where(Tenant.active == True, Tenant.operator_chat_id != None)
        )
        tenants = result.scalars().all()

    for tenant in tenants:
        try:
            tg_app = telegram_apps.get(tenant.bot_token)
            if not tg_app:
                continue
            async with AsyncSessionLocal() as db:
                result = await db.execute(
                    select(UnansweredQuery.question, func.count().label("cnt"))
                    .where(
                        UnansweredQuery.tenant_id == tenant.id,
                        UnansweredQuery.created_at >= cutoff,
                    )
                    .group_by(UnansweredQuery.question)
                    .order_by(text("cnt DESC"))
                    .limit(5)
                )
                rows = result.fetchall()
            if not rows:
                continue
            lines = ["📊 *Consultas sin respuesta (últimas 24h):*\n"]
            for i, row in enumerate(rows, 1):
                lines.append(f"{i}. {row.question} ({row.cnt}×)")
            await tg_app.bot.send_message(
                chat_id=tenant.operator_chat_id,
                text="\n".join(lines),
                parse_mode="Markdown",
            )
        except Exception:
            logger.exception("daily_digest_job failed for tenant %s", tenant.slug)


async def cleanup_job():
    cutoff = datetime.now(timezone.utc) - timedelta(days=90)
    async with AsyncSessionLocal() as db:
        await db.execute(
            text("DELETE FROM unanswered_queries WHERE created_at < :cutoff"),
            {"cutoff": cutoff},
        )
        await db.commit()
    logger.info("cleanup_job: deleted UnansweredQuery rows older than 90 days")


# ─── ngrok URL discovery ──────────────────────────────────────────────────────

async def _get_ngrok_domain(client) -> str | None:
    """Query ngrok's local API (http://ngrok:4040) to get the public HTTPS URL.
    Retries for up to 20 seconds to handle startup ordering."""
    for _ in range(20):
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


# ─── Lifespan ─────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    from rag import http_client

    if settings.sentry_dsn:
        sentry_sdk.init(
            dsn=settings.sentry_dsn,
            integrations=[FastApiIntegration(), SqlalchemyIntegration()],
            traces_sample_rate=0.1,
            environment=settings.environment,
        )

    await init_db()

    # Discover public domain (ngrok in dev, APP_DOMAIN in prod)
    domain = await _get_ngrok_domain(http_client) or settings.app_domain

    # Build Application per active tenant and register webhooks
    from db import AsyncSessionLocal
    async with AsyncSessionLocal() as db:
        result = await db.execute(select(Tenant).where(Tenant.active == True))
        tenants = result.scalars().all()

    for tenant in tenants:
        try:
            tg_app = ApplicationBuilder().token(tenant.bot_token).build()
            _register_handlers(tg_app)
            tg_app.bot_data["tenant"] = tenant
            await tg_app.initialize()
            await tg_app.bot.set_webhook(
                url=f"https://{domain}/webhook/{tenant.slug}",
                secret_token=tenant.webhook_secret,
            )
            telegram_apps[tenant.bot_token] = tg_app
            logger.info("Webhook set for tenant %s", tenant.slug)
        except Exception:
            logger.exception("Failed to initialize tenant %s — skipping", tenant.slug)

    scheduler = AsyncIOScheduler(timezone="UTC", misfire_grace_time=300)
    scheduler.add_job(daily_digest_job, "cron", hour=8, minute=0, id="daily_digest")
    scheduler.add_job(cleanup_job, "cron", day_of_week="sun", hour=0, minute=0, id="weekly_cleanup")
    scheduler.start()

    logger.info("RAG Bot API ready — %d tenant(s) loaded", len(telegram_apps))
    yield

    scheduler.shutdown()

    for tg_app in telegram_apps.values():
        try:
            await tg_app.shutdown()
        except Exception:
            pass
    await http_client.aclose()


# ─── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="RAG Bot API",
    description="Multi-tenant RAG bot platform",
    version="2.0.0",
    lifespan=lifespan,
)
app.state.limiter = limiter

@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(status_code=429, content={"detail": "Rate limit exceeded"})


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "model": settings.llm_model}


@app.post("/upload")
@limiter.limit("10/minute")
async def upload_document(
    request: Request,
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
    tenant: Tenant = Depends(require_tenant),
):
    fname = file.filename.lower()
    if not (fname.endswith(".pdf") or fname.endswith(".md") or fname.endswith(".txt")):
        raise HTTPException(400, "Supported formats: PDF, Markdown (.md), plain text (.txt)")

    namespace = tenant.slug
    content = await file.read()

    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "File too large. Maximum size is 10MB.")

    all_chunks = []

    if fname.endswith(".pdf"):
        try:
            reader = PdfReader(io.BytesIO(content))
        except Exception as e:
            raise HTTPException(400, f"Could not read PDF: {e}")
        for page_num, page in enumerate(reader.pages):
            page_text = page.extract_text() or ""
            if page_text.strip():
                chunks = chunk_text(page_text, source=file.filename, page=page_num + 1)
                all_chunks.extend(chunks)
        pages_processed = len(reader.pages)
    else:
        # Markdown / plain text — treat entire file as a single "page"
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError:
            raise HTTPException(400, "File must be UTF-8 encoded")
        if text.strip():
            all_chunks = chunk_text(text, source=file.filename, page=1)
        pages_processed = 1

    if not all_chunks:
        raise HTTPException(400, "No text could be extracted from this file")

    stored = await index_chunks(db, all_chunks, namespace)

    return {
        "status": "indexed",
        "namespace": namespace,
        "filename": file.filename,
        "pages_processed": pages_processed,
        "chunks_stored": stored,
    }


@app.get("/stats")
@limiter.limit("10/minute")
async def stats(
    request: Request,
    db: AsyncSession = Depends(get_db),
    tenant: Tenant = Depends(require_tenant),
):
    result = await db.execute(
        select(
            DocumentChunk.namespace,
            DocumentChunk.source,
            func.count(DocumentChunk.id).label("chunks"),
        )
        .where(DocumentChunk.namespace == tenant.slug)
        .group_by(DocumentChunk.namespace, DocumentChunk.source)
        .order_by(DocumentChunk.namespace)
    )
    rows = result.fetchall()
    return {
        "indexed_documents": [
            {"namespace": r.namespace, "source": r.source, "chunks": r.chunks}
            for r in rows
        ]
    }


class TenantUpdate(BaseModel):
    expertise_area: str


@app.patch("/tenant")
async def update_tenant(
    body: TenantUpdate,
    db: AsyncSession = Depends(get_db),
    tenant: Tenant = Depends(require_tenant),
):
    tenant.expertise_area = body.expertise_area
    db.add(tenant)
    await db.commit()
    await db.refresh(tenant)

    # Update the in-memory cached tenant so the bot uses the new value immediately
    tg_app = telegram_apps.get(tenant.bot_token)
    if tg_app:
        tg_app.bot_data["tenant"] = tenant

    return {"status": "updated", "expertise_area": tenant.expertise_area}


@app.delete("/namespace")
async def delete_namespace(
    db: AsyncSession = Depends(get_db),
    tenant: Tenant = Depends(require_tenant),
):
    ns = tenant.slug
    await db.execute(text("DELETE FROM document_chunks WHERE namespace = :ns"), {"ns": ns})
    await db.execute(text("DELETE FROM conversations WHERE namespace = :ns"), {"ns": ns})
    await db.commit()
    return {"status": "deleted", "namespace": ns}


@app.post("/webhook/{tenant_slug}")
@limiter.limit("20/minute")
async def telegram_webhook(
    tenant_slug: str,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    received = request.headers.get("X-Telegram-Bot-Api-Secret-Token", "")

    result = await db.execute(
        select(Tenant).where(Tenant.slug == tenant_slug, Tenant.active == True)
    )
    tenant = result.scalar_one_or_none()
    if not tenant:
        raise HTTPException(status_code=404, detail="Tenant not found")

    if not hmac.compare_digest(received, tenant.webhook_secret):
        raise HTTPException(status_code=403, detail="Invalid webhook signature")

    tg_app = telegram_apps.get(tenant.bot_token)
    if not tg_app:
        logger.error("No Application for tenant %s — restart may be needed", tenant_slug)
        return {"ok": True}

    data = await request.json()
    update = Update.de_json(data, tg_app.bot)
    await tg_app.process_update(update)
    return {"ok": True}


# ─── Admin UI ────────────────────────────────────────────────────────────────

VALID_PLANS = ("free", "basic", "pro")

_basic = HTTPBasic()


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


def _admin_html(tenants: list, message: str = "", error: str = "", new_api_key: str = "") -> str:
    rows = ""
    for t in tenants:
        rows += f"""
        <tr>
          <td>{t.id}</td>
          <td><strong>{t.slug}</strong></td>
          <td>{"✅ activo" if t.active else "❌ inactivo"}</td>
          <td>{t.plan}</td>
          <td>
            <form method="post" action="/admin/tenant/{t.id}" style="display:grid;gap:6px">
              <input name="expertise_area" value="{t.expertise_area or ''}" placeholder="área de expertise"
                     style="padding:4px 8px;border:1px solid #ccc;border-radius:4px">
              <input name="contact_url" value="{t.contact_url or ''}" placeholder="https://... (URL contacto)"
                     style="padding:4px 8px;border:1px solid #ccc;border-radius:4px">
              <input name="operator_chat_id" value="{t.operator_chat_id or ''}" placeholder="chat_id operador"
                     style="padding:4px 8px;border:1px solid #ccc;border-radius:4px">
              <button type="submit" style="padding:4px 12px;background:#2563eb;color:#fff;border:none;border-radius:4px;cursor:pointer">
                Guardar
              </button>
            </form>
          </td>
          <td><a href="/admin/queries/{t.id}" style="color:#2563eb;font-size:0.85rem">Ver consultas</a></td>
        </tr>"""

    msg_html = f'<div class="alert ok">{message}</div>' if message else ""
    err_html = f'<div class="alert err">{error}</div>' if error else ""
    key_html = f"""
      <div class="alert key">
        <strong>API Key generada (guardala, no se vuelve a mostrar):</strong><br>
        <code style="font-size:0.95rem;word-break:break-all">{new_api_key}</code>
      </div>""" if new_api_key else ""

    return f"""<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="utf-8">
  <title>Admin — RAG Bot</title>
  <style>
    body {{ font-family: system-ui, sans-serif; max-width: 1000px; margin: 40px auto; padding: 0 20px; color: #1e293b; }}
    h1 {{ font-size: 1.5rem; margin-bottom: 4px; }}
    h2 {{ font-size: 1.1rem; margin: 32px 0 12px; }}
    p.sub {{ color: #64748b; margin-bottom: 24px; }}
    table {{ width: 100%; border-collapse: collapse; margin-bottom: 40px; }}
    th {{ text-align: left; padding: 8px 12px; background: #f1f5f9; border-bottom: 2px solid #e2e8f0; }}
    td {{ padding: 10px 12px; border-bottom: 1px solid #e2e8f0; vertical-align: middle; }}
    tr:hover td {{ background: #f8fafc; }}
    .alert {{ padding: 12px 16px; border-radius: 6px; margin-bottom: 16px; }}
    .ok  {{ background: #dcfce7; color: #166534; border: 1px solid #86efac; }}
    .err {{ background: #fee2e2; color: #991b1b; border: 1px solid #fca5a5; }}
    .key {{ background: #fef9c3; color: #713f12; border: 1px solid #fde047; }}
    .card {{ background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 20px 24px; margin-bottom: 32px; }}
    .row {{ display:flex; gap:12px; flex-wrap:wrap; margin-bottom:12px; }}
    label {{ display:block; font-size:0.85rem; font-weight:600; margin-bottom:4px; color:#475569; }}
    input[type=text] {{ padding:6px 10px; border:1px solid #cbd5e1; border-radius:4px; width:100%; box-sizing:border-box; }}
    .field {{ flex:1; min-width:180px; }}
    .btn {{ padding:8px 20px; background:#2563eb; color:#fff; border:none; border-radius:4px; cursor:pointer; font-size:0.9rem; }}
    .btn:hover {{ background:#1d4ed8; }}
  </style>
</head>
<body>
  <h1>RAG Bot — Panel de Administración</h1>
  <p class="sub">Gestioná los tenants del sistema.</p>

  {msg_html}{err_html}{key_html}

  <h2>➕ Nuevo tenant</h2>
  <div class="card">
    <form method="post" action="/admin/tenants">
      <div class="row">
        <div class="field">
          <label for="slug">Slug <small>(identificador único, sin espacios)</small></label>
          <input type="text" id="slug" name="slug" placeholder="mi-empresa" required>
        </div>
        <div class="field">
          <label for="bot_token">Telegram Bot Token</label>
          <input type="text" id="bot_token" name="bot_token" placeholder="123456:ABC..." required>
        </div>
      </div>
      <div class="row">
        <div class="field">
          <label for="expertise_area">Área de expertise</label>
          <input type="text" id="expertise_area" name="expertise_area" placeholder="servicios de salud y seguros">
        </div>
        <div class="field">
          <label for="plan">Plan</label>
          <select id="plan" name="plan" style="padding:6px 10px;border:1px solid #cbd5e1;border-radius:4px;width:100%">
            <option value="free">Free</option>
            <option value="basic">Basic</option>
            <option value="pro">Pro</option>
          </select>
        </div>
      </div>
      <div class="row">
        <div class="field">
          <label for="contact_url">URL de contacto <small>(http/https)</small></label>
          <input type="text" id="contact_url" name="contact_url" placeholder="https://wa.me/549...">
        </div>
        <div class="field">
          <label for="operator_chat_id">Chat ID del operador <small>(para digest diario)</small></label>
          <input type="text" id="operator_chat_id" name="operator_chat_id" placeholder="123456789">
        </div>
      </div>
      <div class="row">
        <div class="field" style="flex:2">
          <label for="example_questions">Preguntas de ejemplo <small>(una por línea, máx 5)</small></label>
          <textarea id="example_questions" name="example_questions" rows="4"
                    style="padding:6px 10px;border:1px solid #cbd5e1;border-radius:4px;width:100%;box-sizing:border-box"
                    placeholder="¿Cuáles son los horarios?&#10;¿Qué planes tienen?&#10;¿Cómo me registro?"></textarea>
        </div>
      </div>
      <button type="submit" class="btn">Crear tenant</button>
    </form>
  </div>

  <h2>📋 Tenants existentes</h2>
  <table>
    <thead><tr><th>ID</th><th>Slug</th><th>Estado</th><th>Plan</th><th>Configuración</th><th>Consultas</th></tr></thead>
    <tbody>{rows}</tbody>
  </table>
</body>
</html>"""


async def _all_tenants(db: AsyncSession) -> list:
    result = await db.execute(select(Tenant).order_by(Tenant.id))
    return result.scalars().all()


async def _init_tenant_bot(tenant: Tenant, domain: str) -> bool:
    """Build telegram Application for a tenant and register its webhook."""
    try:
        tg_app = ApplicationBuilder().token(tenant.bot_token).build()
        _register_handlers(tg_app)
        tg_app.bot_data["tenant"] = tenant
        await tg_app.initialize()
        await tg_app.bot.set_webhook(
            url=f"https://{domain}/webhook/{tenant.slug}",
            secret_token=tenant.webhook_secret,
        )
        telegram_apps[tenant.bot_token] = tg_app
        return True
    except Exception:
        logger.exception("Failed to initialize bot for tenant %s", tenant.slug)
        return False


@app.get("/admin", response_class=HTMLResponse, include_in_schema=False)
async def admin_panel(
    db: AsyncSession = Depends(get_db),
    _: None = Depends(_require_admin),
):
    return _admin_html(await _all_tenants(db))


@app.post("/admin/tenants", response_class=HTMLResponse, include_in_schema=False)
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
        return _admin_html(await _all_tenants(db), error="Slug y Bot Token son obligatorios.")

    if plan not in VALID_PLANS:
        return _admin_html(await _all_tenants(db), error=f"Plan inválido '{plan}'. Opciones: {', '.join(VALID_PLANS)}.")

    if contact_url and not contact_url.startswith(("http://", "https://")):
        return _admin_html(await _all_tenants(db), error="URL de contacto debe empezar con http:// o https://")

    existing = await db.execute(select(Tenant).where(Tenant.slug == slug))
    if existing.scalar_one_or_none():
        return _admin_html(await _all_tenants(db), error=f"Ya existe un tenant con slug '{slug}'.")

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

    # Discover current public domain from ngrok or settings
    from rag import http_client
    domain = None
    try:
        resp = await http_client.get("http://ngrok:4040/api/tunnels", timeout=2.0)
        for t in resp.json().get("tunnels", []):
            if t.get("proto") == "https":
                domain = t["public_url"].replace("https://", "")
                break
    except Exception:
        pass
    domain = domain or settings.app_domain

    ok = await _init_tenant_bot(tenant, domain)
    if not ok:
        return _admin_html(
            await _all_tenants(db),
            error=f"Tenant '{slug}' creado en DB pero falló al inicializar el bot. Revisá el Bot Token.",
            new_api_key=raw_api_key,
        )

    logger.info("Admin created tenant %s", slug)
    return _admin_html(
        await _all_tenants(db),
        message=f"✓ Tenant '{slug}' creado y bot registrado.",
        new_api_key=raw_api_key,
    )


@app.post("/admin/tenant/{tenant_id}", response_class=HTMLResponse, include_in_schema=False)
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

    if contact_url and not contact_url.startswith(("http://", "https://")):
        result = await db.execute(select(Tenant).order_by(Tenant.id))
        tenants = result.scalars().all()
        return _admin_html(tenants, error="URL de contacto debe empezar con http:// o https://")

    result = await db.execute(select(Tenant).where(Tenant.id == tenant_id))
    tenant = result.scalar_one_or_none()
    if not tenant:
        raise HTTPException(404, "Tenant not found")

    tenant.expertise_area = expertise_area
    tenant.contact_url = contact_url
    tenant.operator_chat_id = operator_chat_id
    db.add(tenant)
    await db.commit()
    await db.refresh(tenant)

    tg_app = telegram_apps.get(tenant.bot_token)
    if tg_app:
        tg_app.bot_data["tenant"] = tenant

    return _admin_html(await _all_tenants(db), message=f"✓ Tenant '{tenant.slug}' actualizado.")


@app.get("/admin/queries/{tenant_id}", response_class=HTMLResponse, include_in_schema=False)
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

    rows_html = ""
    for q in queries:
        ts = q.created_at.strftime("%Y-%m-%d %H:%M") if q.created_at else ""
        rows_html += (
            f"<tr><td>{ts}</td><td>{q.intent_category}</td>"
            f"<td>{q.user_id}</td><td>{q.question}</td></tr>"
        )

    return f"""<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="utf-8">
  <title>Consultas — {tenant.slug}</title>
  <style>
    body {{ font-family: system-ui, sans-serif; max-width: 1100px; margin: 40px auto; padding: 0 20px; color: #1e293b; }}
    h1 {{ font-size: 1.4rem; margin-bottom: 4px; }}
    a {{ color: #2563eb; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 24px; }}
    th {{ text-align: left; padding: 8px 12px; background: #f1f5f9; border-bottom: 2px solid #e2e8f0; }}
    td {{ padding: 8px 12px; border-bottom: 1px solid #e2e8f0; vertical-align: top; word-break: break-word; }}
    tr:hover td {{ background: #f8fafc; }}
  </style>
</head>
<body>
  <h1>Consultas sin respuesta — <em>{tenant.slug}</em></h1>
  <a href="/admin">← Volver al panel</a>
  <table>
    <thead><tr><th>Fecha</th><th>Intent</th><th>User ID</th><th>Pregunta</th></tr></thead>
    <tbody>{rows_html}</tbody>
  </table>
</body>
</html>"""


# ─── Bot handler registration (imported by lifespan) ──────────────────────────

def _register_handlers(tg_app):
    from bot import cmd_start, cmd_help, cmd_sources, cmd_clear, cmd_contactar, handle_message
    from telegram.ext import CommandHandler, MessageHandler, filters
    tg_app.add_handler(CommandHandler("start", cmd_start))
    tg_app.add_handler(CommandHandler("help", cmd_help))
    tg_app.add_handler(CommandHandler("sources", cmd_sources))
    tg_app.add_handler(CommandHandler("clear", cmd_clear))
    tg_app.add_handler(CommandHandler("contactar", cmd_contactar))
    tg_app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
