"""
Database models and connection setup.
Uses pgvector for similarity search on document embeddings.
"""
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Boolean, Column, DateTime, ForeignKey, Index,
    Integer, JSON, String, Text, UniqueConstraint, func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy import text

from config import settings

Base = declarative_base()


class Tenant(Base):
    __tablename__ = "tenants"

    id = Column(Integer, primary_key=True)
    slug = Column(String(64), unique=True, nullable=False, index=True)
    api_key_hash = Column(String(64), unique=True, nullable=False)
    webhook_secret = Column(String(64), nullable=False)
    bot_token = Column(String(128), unique=True, nullable=False)
    plan = Column(String(32), default="free")
    billing_id = Column(String(128), nullable=True)
    expertise_area = Column(String(255), nullable=True, default="")
    contact_url = Column(String(512), nullable=True)
    example_questions = Column(JSON, nullable=True)
    operator_chat_id = Column(String(64), nullable=True)
    created_at = Column(DateTime, server_default=func.now())
    active = Column(Boolean, default=True)

    # Vision / web search
    web_search_enabled = Column(Boolean, default=False, server_default="false")

    # E4/E5: Document structure summary (generated at index time)
    doc_structure_summary = Column(Text, nullable=True)

    # WhatsApp / multi-channel
    wa_phone_number_id = Column(String(100), nullable=True)
    _wa_access_token = Column("wa_access_token", String(500), nullable=True)
    _wa_app_secret = Column("wa_app_secret", String(100), nullable=True)
    wa_business_id = Column(String(100), nullable=True)
    wa_verify_token = Column(String(100), nullable=True)
    wa_reengagement_template = Column(String(200), nullable=True)
    channels = Column(Text, nullable=True, server_default="telegram")

    # Portal (PR1): bcrypt-hashed portal password for tenant self-service login
    portal_password_hash = Column(String(128), nullable=True)

    @hybrid_property
    def wa_access_token(self):
        """Decrypt WA access token on read. Falls back to plaintext if ENCRYPTION_KEY not set."""
        from crypto import decrypt_value
        return decrypt_value(self._wa_access_token) if self._wa_access_token else self._wa_access_token

    @wa_access_token.setter
    def wa_access_token(self, value):
        """Encrypt WA access token on write. Stores plaintext if ENCRYPTION_KEY not set."""
        from crypto import encrypt_value
        self._wa_access_token = encrypt_value(value) if value else value

    @hybrid_property
    def wa_app_secret(self):
        """Decrypt WA app secret on read. Falls back to plaintext if ENCRYPTION_KEY not set."""
        from crypto import decrypt_value
        return decrypt_value(self._wa_app_secret) if self._wa_app_secret else self._wa_app_secret

    @wa_app_secret.setter
    def wa_app_secret(self, value):
        """Encrypt WA app secret on write. Stores plaintext if ENCRYPTION_KEY not set."""
        from crypto import encrypt_value
        self._wa_app_secret = encrypt_value(value) if value else value


class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    id = Column(Integer, primary_key=True, autoincrement=True)
    namespace = Column(String(100), nullable=False, index=True)
    source = Column(String(255), nullable=False)
    page = Column(Integer, default=0)
    content = Column(Text, nullable=False)
    embedding = Column(Vector(settings.embedding_dim))
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    # E4: Chunk metadata for multi-tenant RAG generalization
    chunk_type = Column(String(20), nullable=True)  # price_row, faq_answer, policy_statement, section_header, general_info
    metadata_ = Column("metadata", JSONB, nullable=True, server_default="{}")  # section_name, section_emoji, etc.
    # Note: migration c5d6e7f8g901 converts this column to JSONB at the DB level.
    # SQLAlchemy JSON type works with both; the actual column type is JSONB in production.

    __table_args__ = (
        Index(
            "ix_embedding_hnsw",
            embedding,
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 128},
        ),
        Index("ix_document_chunks_namespace_type", "namespace", "chunk_type"),
    )


class UnansweredQuery(Base):
    __tablename__ = "unanswered_queries"

    id = Column(Integer, primary_key=True, autoincrement=True)
    tenant_id = Column(Integer, ForeignKey("tenants.id"), nullable=True)
    namespace = Column(String(128), nullable=False)
    question = Column(Text, nullable=False)
    user_id = Column(String(64), nullable=False)
    intent_category = Column(String(64), nullable=False)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    __table_args__ = (
        Index("ix_unanswered_queries_tenant_created", "tenant_id", "created_at"),
        Index("ix_unanswered_queries_created_at", "created_at"),
    )


class Conversation(Base):
    __tablename__ = "conversations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    tenant_id = Column(Integer, ForeignKey("tenants.id"), nullable=True)
    user_id = Column(String(64), nullable=False, index=True)
    channel = Column(String(20), default="telegram")
    namespace = Column(String(128), nullable=False)
    role = Column(String(20), nullable=False)
    content = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    __table_args__ = (
        Index("ix_conversations_ns_user_created", "namespace", "user_id", "created_at"),
    )


class SystemConfig(Base):
    """Encrypted key-value store for system-level settings (LLM, embeddings, etc.).
    Values are encrypted with Fernet (ENCRYPTION_KEY env var).
    """
    __tablename__ = "system_config"

    id = Column(Integer, primary_key=True, autoincrement=True)
    key = Column(String(100), unique=True, nullable=False, index=True)
    encrypted_value = Column(Text, nullable=False)
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc),
                        onupdate=lambda: datetime.now(timezone.utc))


class WaServiceWindow(Base):
    __tablename__ = "wa_service_windows"

    id = Column(Integer, primary_key=True, autoincrement=True)
    tenant_id = Column(Integer, ForeignKey("tenants.id"), nullable=False)
    user_id = Column(String(100), nullable=False)
    last_user_message_at = Column(DateTime(timezone=True), nullable=False)

    __table_args__ = (
        UniqueConstraint("tenant_id", "user_id", name="uq_wa_window_tenant_user"),
    )


class Feedback(Base):
    """User feedback (thumbs up/down) on bot responses."""
    __tablename__ = "feedback"

    id = Column(Integer, primary_key=True, autoincrement=True)
    tenant_id = Column(Integer, ForeignKey("tenants.id"), nullable=True)
    user_id = Column(String(64), nullable=False, index=True)
    namespace = Column(String(128), nullable=False)
    message_id = Column(String(64), nullable=True)   # TG message_id or WA message ID
    rating = Column(String(16), nullable=False)        # "positive" or "negative"
    comment = Column(Text, nullable=True)              # optional follow-up text
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    __table_args__ = (
        Index("ix_feedback_ns_created", "namespace", "created_at"),
    )


class TenantAuditLog(Base):
    """Per-tenant knowledge mutation audit trail (E6)."""
    __tablename__ = "tenant_audit_log"

    id = Column(Integer, primary_key=True, autoincrement=True)
    tenant_id = Column(Integer, ForeignKey("tenants.id"), nullable=False, index=True)
    namespace = Column(String(128), nullable=False)
    actor = Column(String(64), nullable=False)      # "tenant:slug", "admin", "system"
    action = Column(String(64), nullable=False)      # "login", "document.upload", etc.
    detail = Column(JSON, nullable=True)             # arbitrary payload
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    __table_args__ = (
        Index("ix_audit_tenant_created", "tenant_id", "created_at"),
        Index("ix_audit_namespace_created", "namespace", "created_at"),
    )


class TenantUsage(Base):
    """Monthly metric counters for E2 usage metering. Implicit reset per month."""
    __tablename__ = "tenant_usage"

    id = Column(Integer, primary_key=True, autoincrement=True)
    tenant_id = Column(Integer, ForeignKey("tenants.id"), nullable=False)
    period_year = Column(Integer, nullable=False)
    period_month = Column(Integer, nullable=False)   # 1-12
    metric = Column(String(64), nullable=False)       # "queries", "uploads", "tokens"
    value = Column(Integer, nullable=False, default=0, server_default="0")

    __table_args__ = (
        UniqueConstraint("tenant_id", "period_year", "period_month", "metric",
                         name="uq_tenant_usage_period_metric"),
        Index("ix_tenant_usage_period", "tenant_id", "period_year", "period_month"),
    )


# ─── Engine + Session ────────────────────────────────────────────────────────

# Admin engine — connects as table owner (ragbot), bypasses RLS.
# Used by /admin routes, lifespan, background jobs, and any cross-tenant path.
engine = create_async_engine(
    settings.database_url,
    echo=False,
    pool_size=10,
    max_overflow=20,
)

AsyncSessionLocal = sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)

# Tenant engine — connects as ragbot_tenant (restricted role, subject to RLS).
# Used by webhook bot path, portal, and tenant-scoped API routes.
# Falls back to admin engine if TENANT_DB_PASSWORD not configured (dev mode).
def _build_tenant_engine():
    if not settings.tenant_db_password:
        # No restricted role configured — reuse admin engine (RLS not enforced).
        # This is fine for local dev without RLS; production MUST set TENANT_DB_PASSWORD.
        return engine
    # Use SQLAlchemy's make_url to parse the connection URL safely.
    # This handles passwords containing @, :, /, and other special characters
    # that would break naive string manipulation.
    from sqlalchemy.engine import make_url
    parsed = make_url(settings.database_url)
    tenant_url = parsed.set(
        username="ragbot_tenant",
        password=settings.tenant_db_password,
    )
    return create_async_engine(
        tenant_url,
        echo=False,
        pool_size=10,
        max_overflow=20,
    )

tenant_engine = _build_tenant_engine()

TenantSessionLocal = sessionmaker(
    tenant_engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


@asynccontextmanager
async def tenant_session(slug: str):
    """Open a tenant-scoped DB session with RLS GUC set.

    Uses SET (not SET LOCAL) so the GUC persists across commits within the
    same session — portal routes that do multiple writes (upload, audit,
    metering) need the tenant scope to survive past the first commit.
    The GUC is explicitly reset on exit to prevent leaking to the pool.
    """
    async with TenantSessionLocal() as session:
        await session.execute(
            text("SELECT set_config('app.current_tenant', :slug, false)"),
            {"slug": slug},
        )
        try:
            yield session
        finally:
            # Explicitly reset the GUC to prevent leaking to the session pool.
            # If the session is broken, this is a no-op — the pool discards it.
            try:
                await session.execute(text("RESET app.current_tenant"))
            except Exception:
                pass  # Session may be broken; pool will discard it


async def get_db():
    """FastAPI dependency: admin DB session (bypasses RLS)."""
    async with AsyncSessionLocal() as session:
        yield session


async def get_tenant_db():
    """FastAPI dependency: tenant-scoped DB session.

    The caller (typically require_tenant_session) must set the RLS GUC
    before any queries. This yields an unscoped TenantSessionLocal session.
    """
    async with TenantSessionLocal() as session:
        yield session


async def init_db():
    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
