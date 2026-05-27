"""baseline

Revision ID: af07e4f0ea14
Revises: 
Create Date: 2026-05-25 18:30:16.688953

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'af07e4f0ea14'
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # IF NOT EXISTS throughout: makes this baseline idempotent for databases
    # that already had tables created by the pre-Alembic Base.metadata.create_all().
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.execute("""
        CREATE TABLE IF NOT EXISTS document_chunks (
            id SERIAL NOT NULL,
            namespace VARCHAR(100) NOT NULL,
            source VARCHAR(255) NOT NULL,
            page INTEGER,
            content TEXT NOT NULL,
            embedding vector(1536),
            created_at TIMESTAMP WITH TIME ZONE,
            PRIMARY KEY (id)
        )
    """)
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_document_chunks_namespace ON document_chunks (namespace)"
    )

    op.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            id SERIAL NOT NULL,
            telegram_user_id VARCHAR(50) NOT NULL,
            namespace VARCHAR(100) NOT NULL,
            role VARCHAR(20) NOT NULL,
            content TEXT NOT NULL,
            created_at TIMESTAMP WITH TIME ZONE,
            PRIMARY KEY (id)
        )
    """)
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_conversations_telegram_user_id ON conversations (telegram_user_id)"
    )


def downgrade() -> None:
    op.drop_table('conversations')
    op.drop_table('document_chunks')
