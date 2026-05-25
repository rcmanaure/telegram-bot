"""
Database models and connection setup.
Uses pgvector for similarity search on document embeddings.
"""
from datetime import datetime, timezone

from pgvector.sqlalchemy import Vector
from sqlalchemy import Column, DateTime, Index, Integer, String, Text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from config import settings

Base = declarative_base()


class DocumentChunk(Base):
    """
    Stores text chunks from uploaded documents alongside their embeddings.
    Each chunk is a ~500 char slice of the original document.
    """
    __tablename__ = "document_chunks"

    id = Column(Integer, primary_key=True, autoincrement=True)
    namespace = Column(String(100), nullable=False, index=True)  # e.g. "acme_company"
    source = Column(String(255), nullable=False)                 # original filename
    page = Column(Integer, default=0)                            # page number in PDF
    content = Column(Text, nullable=False)                       # the actual text chunk
    embedding = Column(Vector(settings.embedding_dim))           # 1536-dim vector
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    # IVFFLAT index for fast approximate nearest-neighbor search
    # For production with >100k rows, switch to HNSW:
    # __table_args__ = (Index("ix_embedding_hnsw", embedding, postgresql_using="hnsw",
    #                         postgresql_with={"m": 16, "ef_construction": 64}),)


class Conversation(Base):
    """
    Stores conversation history per Telegram user.
    Used to maintain context across messages.
    """
    __tablename__ = "conversations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    telegram_user_id = Column(String(50), nullable=False, index=True)
    namespace = Column(String(100), nullable=False)
    role = Column(String(20), nullable=False)    # "user" | "assistant"
    content = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


# ─── Engine + Session ────────────────────────────────────────────────────────
print("🔌 Setting up database connection...")
print(f"   Database URL: {settings.database_url}")
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
    """FastAPI dependency for DB sessions."""
    async with AsyncSessionLocal() as session:
        yield session


async def init_db():
    """Create tables and enable pgvector extension."""
    async with engine.begin() as conn:
        # Enable pgvector
        await conn.execute(__import__("sqlalchemy").text("CREATE EXTENSION IF NOT EXISTS vector"))
        # Create tables
        await conn.run_sync(Base.metadata.create_all)
    print("✅ Database initialized with pgvector")
