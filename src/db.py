"""
Database models and connection setup.
Uses pgvector for similarity search on document embeddings.
"""
from datetime import datetime, timezone

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Boolean, Column, DateTime, ForeignKey, Index,
    Integer, JSON, String, Text, UniqueConstraint, func,
)
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import declarative_base, sessionmaker

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

    # WhatsApp / multi-channel
    wa_phone_number_id = Column(String(100), nullable=True)
    wa_access_token = Column(String(500), nullable=True)
    wa_app_secret = Column(String(100), nullable=True)
    wa_business_id = Column(String(100), nullable=True)
    wa_verify_token = Column(String(100), nullable=True)
    wa_reengagement_template = Column(String(200), nullable=True)
    channels = Column(Text, nullable=True, server_default="telegram")


class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    id = Column(Integer, primary_key=True, autoincrement=True)
    namespace = Column(String(100), nullable=False, index=True)
    source = Column(String(255), nullable=False)
    page = Column(Integer, default=0)
    content = Column(Text, nullable=False)
    embedding = Column(Vector(settings.embedding_dim))
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    __table_args__ = (
        Index(
            "ix_embedding_hnsw",
            embedding,
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
        ),
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


class WaServiceWindow(Base):
    __tablename__ = "wa_service_windows"

    id = Column(Integer, primary_key=True, autoincrement=True)
    tenant_id = Column(Integer, ForeignKey("tenants.id"), nullable=False)
    user_id = Column(String(100), nullable=False)
    last_user_message_at = Column(DateTime(timezone=True), nullable=False)

    __table_args__ = (
        UniqueConstraint("tenant_id", "user_id", name="uq_wa_window_tenant_user"),
    )


# ─── Engine + Session ────────────────────────────────────────────────────────

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


async def get_db():
    async with AsyncSessionLocal() as session:
        yield session


async def init_db():
    async with engine.begin() as conn:
        await conn.execute(__import__("sqlalchemy").text("CREATE EXTENSION IF NOT EXISTS vector"))
