"""
RAG Pipeline: Embed → Store → Retrieve → Answer

This is the core of the demo. Shows clients:
1. How documents get chunked and embedded
2. How semantic search works (not keyword search)
3. How the LLM answers ONLY from retrieved context (no hallucination)
"""
import asyncio
import hashlib
import json
import logging
import re
import time
from collections import OrderedDict
from typing import Awaitable, Callable

import httpx
from services.prompts import build_system_prompt, _GREETING_PATTERN
from services.stt import transcribe_voice
from security import CANARY_TOKEN, scan_chunk_for_injection, sanitize_user_input, validate_output

logger = logging.getLogger(__name__)
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession
from db import AsyncSessionLocal, DocumentChunk, Conversation, UnansweredQuery, Tenant, tenant_session
from sqlalchemy import update
from config import settings
from config_overlay import get_setting, get_setting_int
from llm import (
    call_chat,
    call_chat_with_tools,
    call_embeddings,
    extract_json_from_llm_response,
    is_tool_use_available,
    ToolUseNotSupportedError,
)

http_client = httpx.AsyncClient(timeout=60)


# ─── LLM failover alert to operator ──────────────────────────────────────────

_FAILOVER_ALERT_COOLDOWN = 3600  # seconds (1 hour)
_failover_alert_sent: dict[int, float] = {}  # tenant_id → last alert timestamp


async def _alert_llm_failover(tenant: Tenant) -> None:
    """Send a Telegram alert to the operator when the primary LLM fails over.

    Deduped: at most one alert per tenant per hour. Uses the same
    tg_app.bot.send_message pattern as daily_digest_job.
    """
    if not tenant.operator_chat_id:
        return
    now = time.monotonic()
    last = _failover_alert_sent.get(tenant.id, 0)
    if now - last < _FAILOVER_ALERT_COOLDOWN:
        return
    _failover_alert_sent[tenant.id] = now
    try:
        from state import telegram_apps
        tg_app = telegram_apps.get(tenant.bot_token)
        if not tg_app:
            return
        await tg_app.bot.send_message(
            chat_id=tenant.operator_chat_id,
            text=(
                "⚠️ *Alerta LLM*: el modelo principal falló y se activó el modelo de respaldo.\n"
                "Las respuestas pueden ser más lentas o de menor calidad. "
                "Revisa la configuración del LLM primario cuando puedas."
            ),
            parse_mode="Markdown",
        )
        logger.info("failover_alert_sent tenant=%s chat_id=%s", tenant.slug, tenant.operator_chat_id)
    except Exception:
        logger.warning("failover_alert_failed tenant=%s", tenant.slug, exc_info=True)


# ─── Chunking ────────────────────────────────────────────────────────────────

def _split_markdown_tables(text: str) -> str:
    """Pre-process markdown tables: separate each row into its own paragraph
    so chunk_text treats them as individual units instead of one monolithic block.

    Markdown tables use | as column separators and rows are separated by \\n (not \\n\\n).
    Without this preprocessing, an entire table becomes one paragraph that gets
    character-sliced, diluting the embedding signal for any single row.

    Each data row gets the section header prepended for context:
      "## SISTEMA DIGESTIVO\\n| SDG033 | Apéndice Cecal | $90.00 |"
    """
    lines = text.split('\n')
    result_lines: list[str] = []
    current_header = ""

    def _is_separator(s: str) -> bool:
        return (
            s.startswith('|') and s.endswith('|')
            and set(s.replace('|', '').replace('-', '').replace(':', '')) <= {' '}
        )

    in_table = False
    for i, line in enumerate(lines):
        stripped = line.strip()
        is_table_row = stripped.startswith('|') and stripped.endswith('|')
        is_sep = is_table_row and _is_separator(stripped)

        if stripped.startswith('#'):
            current_header = stripped
            in_table = False

        if is_table_row and not is_sep:
            # Skip column-header rows: the row immediately before a separator row
            # (e.g. "| Código | Descripción | Precio |") has no retrieval value.
            next_nonempty = next((l.strip() for l in lines[i + 1:] if l.strip()), '')
            if _is_separator(next_nonempty):
                if not in_table:
                    if result_lines and result_lines[-1].strip():
                        result_lines.append('')
                    in_table = True
                continue
            if not in_table:
                # Start of a table — insert blank line before for paragraph break
                if result_lines and result_lines[-1].strip():
                    result_lines.append('')
                in_table = True
            # Prepend section header to each row for context
            if current_header:
                result_lines.append(f"{current_header}\n{line}")
            else:
                result_lines.append(line)
            # Blank line after each row → each row is its own paragraph
            result_lines.append('')
        elif is_sep:
            # Skip separator rows (|---|---|...)
            continue
        elif not is_table_row:
            in_table = False
            result_lines.append(line)

    return '\n'.join(result_lines)


def chunk_text(text_content: str, source: str, page: int = 0) -> list[dict]:
    """
    Split text into semantically coherent chunks by splitting on paragraph
    boundaries first, then merging adjacent paragraphs up to chunk_size.
    Falls back to character-based splitting only for oversized paragraphs.
    This keeps section headers with their content so retrieval works correctly.

    Markdown tables are pre-processed so each row becomes its own paragraph
    with the section header prepended — this prevents table rows from being
    character-sliced and ensures each procedure/item gets a distinct embedding.
    """
    # Pre-process markdown tables into row-level paragraphs
    text_content = _split_markdown_tables(text_content)

    size = settings.chunk_size
    overlap = settings.chunk_overlap

    # Split on blank lines to get paragraphs/sections
    paragraphs = [p.strip() for p in text_content.split('\n\n') if p.strip()]

    # For paragraphs that exceed chunk_size, fall back to character slicing
    raw_pieces: list[str] = []
    for para in paragraphs:
        if len(para) <= size:
            raw_pieces.append(para)
        else:
            start = 0
            while start < len(para):
                raw_pieces.append(para[start:start + size])
                start += size - overlap

    # Greedily merge adjacent pieces up to chunk_size.
    # Table row pieces (header + single pipe row) are never merged — each row
    # gets its own chunk so the embedding captures that specific procedure
    # without dilution from 8-9 other rows in the same merged block.
    def _is_table_row_piece(p: str) -> bool:
        lines = p.split('\n')
        return (
            len(lines) == 2
            and lines[0].startswith('#')
            and lines[1].strip().startswith('|')
            and lines[1].strip().endswith('|')
        )

    merged: list[str] = []
    current = ''
    for piece in raw_pieces:
        if not current:
            current = piece
        elif _is_table_row_piece(piece) or _is_table_row_piece(current):
            merged.append(current)
            current = piece
        elif len(current) + 2 + len(piece) <= size:
            current += '\n\n' + piece
        else:
            merged.append(current)
            current = piece
    if current:
        merged.append(current)

    return [
        {"content": content.strip(), "source": source, "page": page}
        for content in merged
        if len(content.strip()) > 50
    ]


# ─── Embeddings (delegated to llm.call_embeddings) ─────────────────────────────


# ─── FAQ chunk sync ──────────────────────────────────────────────────────────

FAQ_SOURCE = "__faq__"


async def sync_faq_chunks(
    db: AsyncSession,
    tenant_slug: str,
    example_questions: list[str] | None,
) -> int:
    """Delete old FAQ chunks and (re)index example_questions for a tenant.
    Atomic: DELETE + INSERT in a single transaction to avoid data loss on failure."""
    await db.execute(
        text("DELETE FROM document_chunks WHERE namespace = :ns AND source = :src"),
        {"ns": tenant_slug, "src": FAQ_SOURCE},
    )

    if not example_questions:
        await db.commit()
        return 0

    chunks = [
        {"content": q.strip(), "source": FAQ_SOURCE, "page": 0}
        for q in example_questions
        if q and q.strip()
    ]
    stored = await index_chunks(db, chunks, tenant_slug, auto_commit=False)
    await db.commit()
    return stored


# ─── Contextual retrieval ────────────────────────────────────────────────────

_CONTEXT_SEMAPHORE = asyncio.Semaphore(5)  # max 5 concurrent LLM context calls


async def _add_contextual_summary(full_doc_text: str, chunk_content: str) -> str:
    """Generate a 1-2 sentence context summary to prepend before embedding.

    The summary situates the chunk within the document so the embedding
    captures section context, not just chunk content.
    Returns "" if LLM_CONTEXT_MODEL is unset or the call fails — caller falls
    back to embedding chunk_content directly.
    """
    context_model = get_setting("llm_context_model", settings.llm_context_model)
    if not context_model:
        return ""

    messages = [
        {
            "role": "user",
            "content": (
                "Dado el siguiente documento completo y un fragmento específico del mismo, "
                "escribe en 1-2 oraciones un contexto breve que sitúe este fragmento dentro "
                "del documento. No repitas el contenido del fragmento. "
                "Responde solo con el contexto, nada más.\n\n"
                f"Documento completo:\n{full_doc_text[:3000]}\n\n"
                f"Fragmento:\n{chunk_content}"
            ),
        }
    ]
    try:
        async with _CONTEXT_SEMAPHORE:
            result = await call_chat(messages, max_tokens=100, temperature=0.0, model=context_model)
        return result.strip() if result else ""
    except Exception as e:
        logger.warning("contextual_summary_failed chunk_preview=%r error=%s", chunk_content[:60], e)
        return ""


# ─── Indexing ────────────────────────────────────────────────────────────────

async def index_chunks(
    db: AsyncSession,
    chunks: list[dict],
    namespace: str,
    auto_commit: bool = True,
    full_doc_text: str | None = None,
) -> int:
    """
    Embed and store chunks in pgvector.
    Returns number of chunks stored.

    When auto_commit=False, the caller is responsible for committing the
    transaction (used for atomic upsert: DELETE old + INSERT new in one commit).

    When full_doc_text is provided and LLM_CONTEXT_MODEL is configured, each
    chunk gets a contextual summary prepended to its embedding input (not stored
    in DB — original content stays clean). FAQ chunks pass full_doc_text=None
    to skip contextual retrieval.
    """
    if not chunks:
        return 0

    clean_chunks = []
    for chunk in chunks:
        if scan_chunk_for_injection(chunk["content"]):
            logger.warning(
                "injection_in_doc source=%s chunk_preview=%r",
                chunk["source"],
                chunk["content"][:80],
            )
            continue
        clean_chunks.append(chunk)

    if not clean_chunks:
        return 0

    # Contextual retrieval: parallel LLM calls outside DB transaction.
    # DB transaction only covers the INSERT below.
    if full_doc_text and get_setting("llm_context_model", settings.llm_context_model):
        summaries = await asyncio.gather(
            *[_add_contextual_summary(full_doc_text, c["content"]) for c in clean_chunks]
        )
        texts = [
            f"{summary}\n{c['content']}" if summary else c["content"]
            for summary, c in zip(summaries, clean_chunks)
        ]
        logger.info(
            "contextual_retrieval ns=%s chunks=%d with_context=%d",
            namespace, len(clean_chunks), sum(1 for s in summaries if s),
        )
    else:
        texts = [c["content"] for c in clean_chunks]

    embeddings = await call_embeddings(texts)

    db_chunks = [
        DocumentChunk(
            namespace=namespace,
            source=chunk["source"],
            page=chunk["page"],
            content=chunk["content"],  # always store original, never contextual text
            embedding=embedding,
        )
        for chunk, embedding in zip(clean_chunks, embeddings)
    ]

    db.add_all(db_chunks)
    if auto_commit:
        await db.commit()

    return len(db_chunks)


# ─── Retrieval ───────────────────────────────────────────────────────────────

async def retrieve_context(
    db: AsyncSession,
    query: str,
    namespace: str,
    top_k: int = None,
) -> list[dict]:
    """
    Find the most semantically similar chunks to the query.
    Uses pgvector's cosine distance (<=> operator).
    """
    top_k = top_k or settings.top_k_results

    # Embed the query
    query_embedding = (await call_embeddings([query]))[0]

    # HNSW tuning: SET LOCAL persists within SQLAlchemy's autobegin transaction.
    # DO NOT insert db.commit() between these SET LOCAL statements and the
    # SELECT below — that would end the implicit transaction and lose the settings.
    ef_search = get_setting_int("hnsw_ef_search", settings.hnsw_ef_search)
    iterative_scan = get_setting("hnsw_iterative_scan", settings.hnsw_iterative_scan)
    if iterative_scan not in ("off", "relaxed_order", "strict_order"):
        logger.warning("invalid hnsw_iterative_scan=%r, using default", iterative_scan)
        iterative_scan = settings.hnsw_iterative_scan
    await db.execute(text(f"SET LOCAL hnsw.ef_search = {int(ef_search)}"))
    await db.execute(text(f"SET LOCAL hnsw.iterative_scan = {iterative_scan}"))

    # pgvector cosine similarity search
    # <=> is cosine distance (lower = more similar)
    result = await db.execute(
        text("""
            SELECT content, source, page,
                   1 - (embedding <=> CAST(:query_vec AS vector)) AS similarity
            FROM document_chunks
            WHERE namespace = :namespace
              AND embedding IS NOT NULL
            ORDER BY embedding <=> CAST(:query_vec AS vector)
            LIMIT :top_k
        """),
        {
            "query_vec": str(query_embedding),
            "namespace": namespace,
            "top_k": top_k,
        }
    )

    rows = result.fetchall()
    return [_row_to_chunk_dict(row, similarity=round(float(row.similarity), 3)) for row in rows]


async def retrieve_catalog_overview(
    db: AsyncSession,
    namespace: str,
) -> list[dict]:
    """Return one representative chunk per catalog section for broad overview queries.

    Prefers typed chunks (chunk_type = 'section_header') for efficiency.
    Falls back to regex heuristic (content ~ '^## ') for existing untyped data.
    """
    # Try typed chunks first (E4 enrichment)
    result = await db.execute(
        text("""
            SELECT content, source, page, chunk_type, metadata
            FROM document_chunks
            WHERE namespace = :namespace
              AND chunk_type = 'section_header'
              AND embedding IS NOT NULL
        """),
        {"namespace": namespace},
    )
    rows = result.fetchall()

    if not rows:
        # Fallback: regex heuristic for pre-existing untyped data
        result = await db.execute(
            text("""
                SELECT DISTINCT ON (substring(content FROM '^## ([^\n]+)'))
                       content, source, page
                FROM document_chunks
                WHERE namespace = :namespace
                  AND content ~ '^## '
                  AND embedding IS NOT NULL
                ORDER BY substring(content FROM '^## ([^\n]+)'), id
            """),
            {"namespace": namespace},
        )
        rows = result.fetchall()

    return [_row_to_chunk_dict(row) for row in rows]


async def retrieve_full_catalog(
    db: AsyncSession,
    namespace: str,
) -> list[dict]:
    """Return ALL catalog-relevant chunks ordered by section for full price-list queries.

    Prefers typed chunks (chunk_type IN ('price_row', 'section_header')) for precision.
    Falls back to regex heuristic (content ~ '^## ') for existing untyped data.
    """
    # Try typed chunks first (E4 enrichment)
    result = await db.execute(
        text("""
            SELECT content, source, page, chunk_type, metadata
            FROM document_chunks
            WHERE namespace = :namespace
              AND chunk_type IN ('price_row', 'section_header')
              AND embedding IS NOT NULL
            ORDER BY
              CASE WHEN chunk_type = 'section_header' THEN 0 ELSE 1 END,
              id
        """),
        {"namespace": namespace},
    )
    rows = result.fetchall()

    if not rows:
        # Fallback: regex heuristic for pre-existing untyped data
        result = await db.execute(
            text("""
                SELECT content, source, page
                FROM document_chunks
                WHERE namespace = :namespace
                  AND content ~ '^## '
                  AND embedding IS NOT NULL
                ORDER BY substring(content FROM '^## ([^\n]+)'), id
            """),
            {"namespace": namespace},
        )
        rows = result.fetchall()

    return [_row_to_chunk_dict(row) for row in rows]


_CATALOG_LLM_PROMPT = """\
Eres un asistente que formatea listas de precios. A continuación recibirás fragmentos
de un catálogo de servicios/productos. Tu trabajo es listar TODOS los ítems exactamente
como aparecen, organizados por sección.

REGLAS INQUEBRANTABLES:
- Lista CADA ítem que aparece en los fragmentos. No omitas ninguno.
- No inventes ítems, precios, ni secciones que no estén en los fragmentos.
- Usa los nombres exactos del documento. No parafrasees ni simplifiques.
- Agrupa por sección cuando los fragmentos tengan encabezados de sección.
- Si un ítem tiene precio, inclúyelo. Si no tiene precio, inclúyelo igual.
- Usa el emoji que aparece en el encabezado de sección si existe. Si no, usa el emoji
  más apropiado para esa sección.
- Formato: *Sección*\n emoji Nombre del ítem — precio\n
- Si el documento no tiene formato de tabla o lista de precios, responde con lo que
  encuentres organizado de la forma más clara posible.

Fragmentos:
{chunks_text}
"""

_CATALOG_RETRY_PROMPT = """\
REINTENTO: Tu respuesta anterior omitió demasiados ítems. Debes incluir TODOS los ítems
de los fragmentos. Lista CADA ítem exactamente como aparece. No omitas ninguno.

Fragmentos:
{chunks_text}
"""

_CATALOG_BATCH_SIZE = 30  # chunks per LLM call for large catalogs
_HALLUCINATION_THRESHOLD = 0.5  # LLM must retain ≥ 50% of source items
_NEUTRAL_SIMILARITY = 0.5  # similarity for policy/section chunks (included by rule, not relevance)
_CLASSIFY_CONCURRENCY = 5  # max parallel chunk classification LLM calls


async def _format_catalog_with_llm(
    chunks: list[dict],
    channel: str = "telegram",
    on_failover: Callable[[], Awaitable[None]] | None = None,
) -> str:
    """Format a complete price list using LLM, with hallucination guard.

    Replaces the old regex-based _format_catalog_as_text with an LLM-first approach
    that works for any document format (not just markdown tables with CODE|DESC|PRICE).

    Item-count verification: if the LLM output has ≤50% of the distinct items found
    in the chunks, retry once with a stricter prompt. If still failing, fall back to
    listing the raw chunks as-is.
    """
    # Count distinct items in source chunks for hallucination guard
    source_item_count = 0
    for c in chunks:
        ct = c.get("chunk_type")
        if ct == "price_row":
            source_item_count += 1
        elif ct == "section_header":
            pass  # headers aren't items
        elif ct is None:
            # Fallback: count lines that look like price rows in untyped chunks
            for line in c["content"].split("\n"):
                if "|" in line and "$" in line:
                    source_item_count += 1

    # For large catalogs, batch the chunks
    if len(chunks) > _CATALOG_BATCH_SIZE:
        batches = [chunks[i:i + _CATALOG_BATCH_SIZE] for i in range(0, len(chunks), _CATALOG_BATCH_SIZE)]
    else:
        batches = [chunks]

    formatted_sections: list[str] = []

    for batch in batches:
        chunks_text = "\n\n---\n\n".join(c["content"] for c in batch)
        messages = [
            {"role": "system", "content": _CATALOG_LLM_PROMPT.format(chunks_text=chunks_text)},
            {"role": "user", "content": "Formatea la lista completa de precios."},
        ]
        result = await call_chat(messages, max_tokens=2000, temperature=0.0,
                                 channel=channel, on_failover=on_failover)

        # Hallucination guard: count items in LLM output vs source
        if source_item_count > 0:
            llm_item_lines = sum(1 for line in result.split("\n")
                                 if line.strip() and not line.strip().startswith("*") and "—" in line)
            if llm_item_lines <= source_item_count * _HALLUCINATION_THRESHOLD:
                logger.warning("catalog_hallucination_guard items=%d llm_items=%d — retrying",
                               source_item_count, llm_item_lines)
                retry_messages = [
                    {"role": "system", "content": _CATALOG_RETRY_PROMPT.format(chunks_text=chunks_text)},
                    {"role": "user", "content": "Formatea la lista completa de precios. No omitas ningún ítem."},
                ]
                result = await call_chat(retry_messages, max_tokens=2000, temperature=0.0,
                                         channel=channel, on_failover=on_failover)
                llm_item_lines = sum(1 for line in result.split("\n")
                                     if line.strip() and not line.strip().startswith("*") and "—" in line)
                if llm_item_lines <= source_item_count * _HALLUCINATION_THRESHOLD:
                    logger.warning("catalog_hallucination_guard still failing items=%d llm_items=%d — raw fallback",
                                   source_item_count, llm_item_lines)
                    formatted_sections.append(_format_catalog_raw(batch))
                    continue

        formatted_sections.append(result)

    if len(formatted_sections) == 1:
        return formatted_sections[0]

    return "\n\n".join(formatted_sections)


def _format_catalog_raw(chunks: list[dict]) -> str:
    """Fallback: list raw chunk content when LLM formatting fails verification."""
    sections: dict[str, list[str]] = {}
    section_order: list[str] = []

    for c in chunks:
        content = c["content"]
        metadata = c.get("metadata") or {}
        ct = c.get("chunk_type")

        if ct == "section_header":
            section_name = metadata.get("section_name", content.strip().lstrip("#").strip())
            if section_name not in sections:
                sections[section_name] = []
                section_order.append(section_name)
            sections[section_name].append(content)
        else:
            # Find section header in content (## ...)
            header = ""
            for line in content.split("\n"):
                if line.startswith("## "):
                    header = line[3:].strip()
                    break
            if header and header not in sections:
                sections[header] = []
                section_order.append(header)
            target = sections.get(header, None)
            if target is not None:
                target.append(content)
            else:
                if "General" not in sections:
                    sections["General"] = []
                    section_order.append("General")
                sections["General"].append(content)

    if not section_order:
        return "No encontré información de precios en los documentos."

    lines = []
    for section in section_order:
        lines.append(f"*{section}*")
        for text in sections[section]:
            lines.append(text.strip())
        lines.append("")

    return "\n".join(lines).rstrip()


# ─── Generation ──────────────────────────────────────────────────────────────

MIN_SIMILARITY = 0.20  # chunks below this threshold are considered off-topic
LOW_MIN_SIMILARITY = 0.10  # second-pass threshold for approximate matches
MAX_CONTEXT_CHUNKS = 30   # cap after all merges to prevent context window blowup
VISION_EXTRACT_MAX_TOKENS = 80  # short enough to avoid prose, enough for comma-separated search terms

# ─── Chunk type classification (E4: multi-tenant RAG generalization) ────────────

ALLOWED_CHUNK_TYPES = frozenset({
    "price_row",       # individual item with price/code (e.g. "| GIN001 | Biopsia ... | $80 |")
    "faq_answer",      # Q&A pair (e.g. "### ¿Cuánto cuesta...?\nRespuesta: ...")
    "policy_statement",  # rules, terms, conditions (e.g. "El pago es obligatorio...")
    "section_header",  # category/section heading (e.g. "## GINECOLÓGICO")
    "general_info",    # everything else (hours, contact info, descriptions)
})

_CLASSIFY_SEMAPHORE = asyncio.Semaphore(_CLASSIFY_CONCURRENCY)  # limit concurrent classification calls
_MAX_SECTION_SIBLINGS = 5  # cap to prevent unbounded OR conditions


def _row_to_chunk_dict(row, similarity: float = _NEUTRAL_SIMILARITY) -> dict:
    """Convert a raw SQL row to a chunk dict with consistent attribute access.

    Uses getattr with defaults for optional columns (chunk_type, metadata_)
    that may be absent in queries that don't select them.
    """
    return {
        "content": row.content,
        "source": row.source,
        "page": row.page,
        "similarity": similarity,
        "chunk_type": getattr(row, "chunk_type", None),
        "metadata": getattr(row, "metadata_", None),
    }


async def classify_chunk_type(chunk_content: str) -> str:
    """Classify a single chunk into one of ALLOWED_CHUNK_TYPES using LLM.

    Returns 'general_info' on any failure (safe default).
    """
    messages = [
        {
            "role": "user",
            "content": (
                "Classify this text chunk as exactly one of:\n"
                "- price_row: an individual item with a price, code, or catalog entry\n"
                "- faq_answer: a question and answer pair\n"
                "- policy_statement: rules, terms, conditions, or requirements\n"
                "- section_header: a category or section heading\n"
                "- general_info: anything else (hours, contact info, descriptions)\n\n"
                "Reply with ONLY the classification word, nothing else.\n\n"
                f"Chunk:\n{chunk_content[:500]}"
            ),
        },
    ]
    try:
        async with _CLASSIFY_SEMAPHORE:
            raw = await call_chat(messages, max_tokens=10, temperature=0.0)
        result = raw.strip().lower()
        if result in ALLOWED_CHUNK_TYPES:
            return result
        logger.warning("classify_chunk_type: unexpected result=%r, defaulting to general_info", result)
        return "general_info"
    except Exception as e:
        logger.warning("classify_chunk_type failed: %s — defaulting to general_info", e)
        return "general_info"


async def classify_chunks_batch(chunks: list[dict]) -> list[str]:
    """Classify a batch of chunks in parallel. Returns a list of chunk_type strings."""
    return await asyncio.gather(*[classify_chunk_type(c["content"]) for c in chunks])


async def generate_section_emoji(section_name: str) -> str:
    """Generate a single emoji for a section name using LLM.

    Returns the emoji character, or '🔬' on failure (safe default).
    """
    messages = [
        {
            "role": "user",
            "content": (
                f"Pick exactly one emoji that best represents this section: {section_name}\n"
                "Reply with only the emoji character, nothing else."
            ),
        },
    ]
    try:
        async with _CLASSIFY_SEMAPHORE:
            raw = await call_chat(messages, max_tokens=5, temperature=0.0)
        emoji = raw.strip()
        # Validate: single emoji or short emoji sequence (some emojis are 2+ chars with ZWJ)
        if emoji and len(emoji) <= 8:
            return emoji
        logger.warning("generate_section_emoji: unexpected result=%r, defaulting to 🔬", emoji)
        return "🔬"
    except Exception as e:
        logger.warning("generate_section_emoji failed: %s — defaulting to 🔬", e)
        return "🔬"


async def generate_doc_structure_summary(
    chunk_types: list[str],
    section_names: list[str],
    total_chunks: int,
) -> str:
    """Generate a 2-3 sentence summary of the document structure for the system prompt.

    Uses pre-computed chunk_type counts and section names (no LLM call needed).
    """
    from collections import Counter
    type_counts = Counter(chunk_types)
    price_count = type_counts.get("price_row", 0)
    faq_count = type_counts.get("faq_answer", 0)
    policy_count = type_counts.get("policy_statement", 0)
    header_count = type_counts.get("section_header", 0)

    parts = []
    if price_count > 0:
        parts.append(f"{price_count} ítems con precio")
    if faq_count > 0:
        parts.append(f"{faq_count} preguntas frecuentes")
    if policy_count > 0:
        parts.append(f"{policy_count} políticas o condiciones")
    if header_count > 0:
        parts.append(f"{header_count} secciones")

    if not parts:
        return f"Documento con {total_chunks} fragmentos de información."

    content_desc = ", ".join(parts)

    if section_names:
        # Up to 5 section names, then "y más"
        if len(section_names) > 5:
            sections_str = ", ".join(section_names[:5]) + ", y más"
        else:
            sections_str = ", ".join(section_names)
        return f"Documento organizado en secciones ({sections_str}). Contiene {content_desc}."
    return f"Documento con {content_desc}."


async def post_index_enrichment(
    db: AsyncSession,
    namespace: str,
    tenant_id: int | None,
    chunks: list[dict],
) -> None:
    """Post-index enrichment: classify chunks, generate emojis, and build doc structure summary.

    Called after index_chunks() completes. Runs as an async background task
    so it doesn't block the upload HTTP response. On failure, chunks remain
    usable with NULL chunk_type (fallback heuristics in query pipeline).
    """
    if not chunks or not tenant_id:
        return

    try:
        # 1. Classify chunk types in parallel
        chunk_types = await classify_chunks_batch(chunks)

        # 2. Extract section headers and generate emojis for them
        section_names: list[str] = []
        section_emojis: dict[str, str] = {}
        for i, chunk in enumerate(chunks):
            content = chunk.get("content", "")
            # Detect section headers: lines starting with ##
            lines = content.split("\n", 1)
            if lines[0].startswith("#"):
                section_name = lines[0].lstrip("#").strip()
                if section_name and section_name not in section_emojis:
                    section_names.append(section_name)
                    emoji = await generate_section_emoji(section_name)
                    section_emojis[section_name] = emoji

        # 3. Update chunks with chunk_type and metadata
        chunk_type_map: dict[str, str] = {}  # content → chunk_type (for dedup)
        for chunk, ctype in zip(chunks, chunk_types):
            chunk_type_map[chunk["content"][:200]] = ctype

        # Build metadata per chunk
        for i, (chunk, ctype) in enumerate(zip(chunks, chunk_types)):
            metadata: dict = {}
            lines = chunk.get("content", "").split("\n", 1)
            if lines[0].startswith("#"):
                section_name = lines[0].lstrip("#").strip()
                metadata["section_name"] = section_name
                if section_name in section_emojis:
                    metadata["section_emoji"] = section_emojis[section_name]

            # Update the chunk row in DB
            await db.execute(
                update(DocumentChunk)
                .where(
                    DocumentChunk.namespace == namespace,
                    DocumentChunk.source == chunk["source"],
                    DocumentChunk.content == chunk["content"],
                )
                .values(
                    chunk_type=ctype,
                    metadata_=metadata if metadata else None,
                )
            )

        # 4. Generate and store doc_structure_summary on Tenant
        doc_summary = generate_doc_structure_summary(chunk_types, section_names, len(chunks))
        if tenant_id:
            await db.execute(
                update(Tenant)
                .where(Tenant.id == tenant_id)
                .values(doc_structure_summary=doc_summary)
            )

        await db.commit()
        logger.info(
            "post_index_enrichment ns=%s: classified %d chunks, %d sections, summary=%r",
            namespace, len(chunks), len(section_names), doc_summary[:80],
        )
    except Exception as e:
        logger.warning("post_index_enrichment failed ns=%s: %s", namespace, e, exc_info=True)
        # Non-critical: chunks are still usable with NULL chunk_type

# ─── Illegible image detection ─────────────────────────────────────────────────

def _illegible_fallback_msg(images: list[dict] | None) -> str:
    """Return the appropriate illegible-image message (singular or plural)."""
    if images and len(images) > 1:
        return (
            "No puedo leer las imágenes. La calidad o resolución puede ser insuficiente. "
            "Intenta enviarlas con mejor iluminación o enfoque, o describe tu consulta por texto."
        )
    return (
        "No puedo leer la imagen. La calidad o resolución puede ser insuficiente. "
        "Intenta enviarla con mejor iluminación o enfoque, o describe tu consulta por texto."
    )


# ─── Query reformulation ──────────────────────────────────────────────────────

async def _reformulate_query(question: str, history: list[dict]) -> str:
    """Rewrite a follow-up question into a standalone query using conversation history.

    Returns the original question if history is empty or reformulation fails.
    Uses a fast, low-token LLM call to resolve pronouns and context-dependent
    references (e.g., "¿cuánto cuesta?" → "¿cuánto cuesta el Plan Pro?").
    """
    if not history:
        return question

    # Build compact history summary (last 3 turns = 6 messages max)
    recent = history[-6:]
    history_lines = []
    for msg in recent:
        role = "User" if msg["role"] == "user" else "Bot"
        content = msg["content"][:200]  # truncate long messages
        history_lines.append(f"{role}: {content}")

    messages = [
        {
            "role": "system",
            "content": (
                "You are a query reformulation assistant. Given a follow-up question and the "
                "conversation history, rewrite the question into a self-contained, standalone query "
                "that preserves the original intent but can be understood without context.\n\n"
                "Rules:\n"
                "- Resolve pronouns and references (e.g., '¿cuánto cuesta?' → '¿cuánto cuesta el Plan Pro?')\n"
                "- Keep the same language as the question\n"
                "- If the question is already standalone, return it unchanged\n"
                "- Output ONLY the reformulated question, nothing else\n"
                "- Do not add information not implied by the conversation"
            ),
        },
        {
            "role": "user",
            "content": f"Conversation:\n{chr(10).join(history_lines)}\n\nFollow-up question: {question}",
        },
    ]

    try:
        reformulated = await call_chat(messages, max_tokens=500, temperature=0.0)
        reformulated = reformulated.strip().strip('"').strip("'")
        # Sanity check: if reformulation is wildly different length, keep original
        if len(reformulated) < len(question) * 0.3 or len(reformulated) > len(question) * 5:
            logger.warning("reformulate_query: result too different, keeping original. q=%r r=%r",
                           question[:60], reformulated[:60])
            return question
        return reformulated
    except Exception as e:
        logger.warning("reformulate_query failed: %s — using original question", e)
        return question


# ─── HyDE (Hypothetical Document Embeddings) ──────────────────────────────────


async def _hyde_query(question: str, expertise_area: str) -> str:
    """Generate a hypothetical catalog/document answer and return it as the search key.

    Bridges patient-language vs catalog-language vocabulary gap by embedding
    what the answer would look like, not the question itself.
    Returns "" on failure or for broad listing queries — caller falls back to original question.

    Listing queries (e.g. "qué tipos de biopsias tienen") must NOT use HyDE: the prompt
    generates a single procedure name which biases the embedding toward one catalog section,
    suppressing all other categories from the top-k results.
    """
    messages = [
        {
            "role": "user",
            "content": (
                f"Eres un especialista en {expertise_area}. Dado el siguiente texto de un cliente, "
                f"escribe el nombre técnico/formal del procedimiento o servicio como aparecería "
                f"en un documento de {expertise_area}. "
                f"Usa nomenclatura técnica formal — NO el lenguaje coloquial del cliente. "
                f"NO inventes precios. NO expliques. Responde SOLO con el nombre técnico "
                f"del procedimiento (1-2 líneas máximo).\n\n"
                f"Texto del cliente: {question}"
            ),
        }
    ]
    try:
        result = await call_chat(messages, max_tokens=300, temperature=0.0)
        result = result.strip()
        if not result or len(result) < 3 or len(result) > 500:
            return ""
        return result
    except Exception as e:
        logger.warning("hyde_query failed: %s — using original query", e)
        return ""


async def retrieve_policy_chunks(db: AsyncSession, namespace: str) -> list[dict]:
    """Fetch ALL policy_statement and section_header chunks for a namespace.

    These chunks are critical for complete answers — they contain requirements,
    conditions, and structural context that semantic search often misses due to
    low similarity to price/procedure queries. By always including them when
    context exists, we ensure the LLM sees requirements like "pago anticipado",
    "traer formulario", phone numbers, etc.

    E7: Always-include policies in retrieval results.
    """
    # Try typed chunks first (E4 enrichment)
    result = await db.execute(
        text("""
            SELECT content, source, page, chunk_type, metadata
            FROM document_chunks
            WHERE namespace = :namespace
              AND chunk_type IN ('policy_statement', 'section_header')
              AND embedding IS NOT NULL
        """),
        {"namespace": namespace},
    )
    rows = result.fetchall()

    if not rows:
        # NULL chunk_type fallback: pre-E4 data — use content patterns.
        # Only include section headers and high-confidence policy patterns.
        # Broad patterns like '%importante%' and '%nota:%' were removed because
        # they over-match non-policy chunks (pricing notes, general descriptions).
        result = await db.execute(
            text("""
                SELECT content, source, page, chunk_type, metadata
                FROM document_chunks
                WHERE namespace = :namespace
                  AND embedding IS NOT NULL
                  AND chunk_type IS NULL
                  AND (
                    content ~ '^## '
                    OR content ILIKE '%requisito%'
                    OR content ILIKE '%condición%'
                    OR content ILIKE '%condiciones%'
                    OR content ILIKE '%política%'
                    OR content ILIKE '%instrucciones%'
                  )
                LIMIT 15
            """),
            {"namespace": namespace},
        )
        rows = result.fetchall()

    return [_row_to_chunk_dict(row) for row in rows]


async def retrieve_section_siblings(
    db: AsyncSession,
    namespace: str,
    matched_chunks: list[dict],
) -> list[dict]:
    """Fetch ALL chunks from the same section as price_row chunks in context.

    When a user asks about a specific procedure (e.g. "biopsia extemporánea"),
    the price_row chunk has high similarity but the requirement/policy chunks
    in the same section may not. This function pulls all chunks sharing the
    same (source, section_name) as any price_row chunk, ensuring the LLM sees
    ALL requirements for the procedure.

    E9: Cross-section retrieval for procedure requirements.
    """

    # Collect unique (source, section_name) pairs from price_row chunks
    sections = set()
    for c in matched_chunks:
        if c.get("chunk_type") != "price_row":
            continue
        src = c.get("source", "")
        meta = c.get("metadata") or {}
        section = meta.get("section_name", "")
        if src and section:
            sections.add((src, section))

    if not sections:
        # Fallback for NULL metadata: extract section from content header
        for c in matched_chunks:
            if c.get("chunk_type") == "price_row" or "\n" in c.get("content", ""):
                src = c.get("source", "")
                content = c.get("content", "")
                # Find section header in content (first line starting with ##)
                first_line = content.split("\n")[0] if "\n" in content else ""
                if first_line.startswith("#"):
                    section = first_line.lstrip("#").strip()
                    if src and section:
                        sections.add((src, section))

    if not sections:
        return []

    # Cap sections to prevent unbounded queries
    sections = set(list(sections)[:_MAX_SECTION_SIBLINGS])

    # Build query for all chunks in those sections
    # Using metadata->>'section_name' for typed chunks
    conditions = []
    params: dict = {"namespace": namespace}
    for i, (src, section) in enumerate(sections):
        params[f"src_{i}"] = src
        params[f"sec_{i}"] = section
        conditions.append(f"(source = :src_{i} AND metadata->>'section_name' = :sec_{i})")

    where_clause = " OR ".join(conditions)

    result = await db.execute(
        text(f"""
            SELECT content, source, page, chunk_type, metadata
            FROM document_chunks
            WHERE namespace = :namespace
              AND ({where_clause})
              AND embedding IS NOT NULL
        """),
        params,
    )
    rows = result.fetchall()

    if not rows and sections:
        # Fallback for NULL metadata: use content-based section matching
        # Fetch all chunks from the same source and filter by section header
        sources = {src for src, _ in sections}
        source_conditions = []
        params2: dict = {"namespace": namespace}
        for i, src in enumerate(sources):
            params2[f"src_{i}"] = src
            source_conditions.append(f"source = :src_{i}")
        source_where = " OR ".join(source_conditions)

        result = await db.execute(
            text(f"""
                SELECT content, source, page, chunk_type, metadata
                FROM document_chunks
                WHERE namespace = :namespace
                  AND ({source_where})
                  AND embedding IS NOT NULL
            """),
            params2,
        )
        all_rows = result.fetchall()

        # Filter to chunks that START with the same section header only.
        # The imprecise substring match (sec in content[:100]) was removed — it
        # produced false positives by matching section names mentioned in passing
        # inside chunks that belong to different sections.
        section_headers = {f"## {sec}" for sec in {s for _, s in sections}} | \
                          {f"# {sec}" for sec in {s for _, s in sections}}
        rows = [
            r for r in all_rows
            if any(r.content.startswith(h) for h in section_headers)
        ]

    return [_row_to_chunk_dict(row) for row in rows]


async def _extract_search_terms_from_images(images: list[dict]) -> tuple[str, str]:
    """Use the vision model to extract key search terms from images.

    When a user sends an image with no caption (or a generic default question),
    the vector search has nothing specific to match against. This function
    calls the vision model with a low-token extraction prompt to identify the
    key terms (e.g., study names, medical terms) that can be used as a
    search query for the RAG vector search.

    Returns (legibility, terms) where legibility is one of:
      - "legible": image fully readable
      - "partial": some parts readable, others not
      - "illegible": cannot read image at all
    And terms is a comma-separated search string (empty if illegible).
    On any failure, returns ("illegible", "").
    """
    vision_model = get_setting("llm_vision_model", settings.llm_vision_model)
    if not vision_model:
        return "illegible", ""

    content: list[dict] = [
        {
            "type": "text",
            "text": (
                "Analizá esta imagen y extraé los términos clave para búsqueda.\n"
                "Respondé SOLO con JSON: "
                '{"legibility": "legible"|"partial"|"illegible", "terms": "término1, término2"}\n'
                "- legible: la imagen se puede leer completamente\n"
                "- partial: algunas partes son legibles, otras no\n"
                "- illegible: no se puede leer nada (borrosa, oscura, etc.)\n"
                "- terms: términos clave separados por comas (vacío si illegible)\n"
                "Ejemplo: {\"legibility\": \"legible\", \"terms\": \"Biopsia de apéndice cecal, Anexo de apéndice cecal\"}"
            ),
        },
    ]
    for img in images:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:{img['mime']};base64,{img['b64']}"},
        })

    messages = [{"role": "user", "content": content}]
    try:
        raw = await call_chat(messages, max_tokens=VISION_EXTRACT_MAX_TOKENS, temperature=0.0, model=vision_model)
        parsed = extract_json_from_llm_response(raw.strip())
        legibility = parsed.get("legibility", "legible")
        if legibility not in {"legible", "partial", "illegible"}:
            legibility = "legible"
        terms = parsed.get("terms", "")
        if not isinstance(terms, str):
            terms = str(terms)
        logger.info("vision_extracted legibility=%s terms=%r", legibility, terms[:120])
        return legibility, terms
    except Exception as e:
        logger.warning("vision_extract_failed: %s — fallback illegible", e)
        return "illegible", ""


# ─── LLM calls (delegated to llm.call_chat) ──────────────────────────────────


async def _triage_response(
    question: str,
    expertise_area: str,
    language_code: str | None = None,
    example_questions: list[str] | None = None,
) -> tuple[str, str]:
    """Classify intent and generate fallback reply when no context found.
    Returns (intent, reply_text). Intent: greeting | off_topic | needs_human | ambiguous.
    When example_questions is provided, includes them so the LLM can suggest specific topics."""
    area = expertise_area or "los temas cubiertos en los documentos"
    questions_hint = ""
    if example_questions:
        q_list = ", ".join(f'"{q}"' for q in example_questions[:5])
        questions_hint = f"\n\nThe service can help with topics like: {q_list}. When intent is 'ambiguous', suggest these topics."
    messages = [
        {
            "role": "system",
            "content": (
                f"You are a routing assistant for a service specialized in: {area}.\n"
                "The user sent a message but no relevant documents were found.\n"
                "Your job: classify the intent and write a short, helpful reply (1-2 sentences).\n\n"
                "REPLY RULES — follow exactly:\n"
                "- Do NOT introduce yourself or mention your name.\n"
                "- Do NOT start with greetings ('Hola', 'Hi', '¡Hola!', etc.).\n"
                "- Do NOT ask 'How can I help you?' or similar open-ended questions.\n"
                "- Answer directly and concisely.\n"
                "- Respond in the same language the user wrote in.\n"
                "- When responding in Spanish, use neutral Latin American Spanish: use 'tú/usted/ustedes', "
                "never 'vosotros'. Avoid Spain-specific vocabulary.\n"
                "CRITICAL RULE: If the question mentions ANY procedure, study, price, or service "
                "that could plausibly be offered by the expertise area — classify as 'ambiguous', NEVER 'off_topic'. "
                "off_topic is ONLY for questions with zero possible connection (cooking recipes, sports scores, software bugs).\n\n"
                "IMPORTANT: Respond with ONLY a JSON object, no markdown, no preamble:\n"
                '{"intent": "<greeting|off_topic|needs_human|ambiguous>", "reply": "<reply text>"}\n\n'
                "Intent definitions:\n"
                "- greeting: purely social/phatic messages only (hi, hello, thanks, bye, how are you). "
                "NOT questions about capabilities or what you can do.\n"
                "- ambiguous: question that COULD relate to the area but no info was found — "
                "tell the user what topics you can help with and invite them to ask more specifically. "
                "This includes questions ABOUT the expertise area when no specific document matches.\n"
                "- off_topic: question clearly and obviously unrelated to your area of expertise. "
                "When in doubt, choose 'ambiguous' over 'off_topic'. "
                "For off_topic replies: be warm and redirect politely. Say what you DO cover, "
                "not just what you don't. Never be dismissive or blunt.\n"
                "- needs_human: user explicitly wants to speak with a real person.\n\n"
                "Examples:\n"
                "'hi' → greeting\n"
                "'que planes tienes?' → ambiguous (could be about service plans)\n"
                "'que puedes hacer?' → ambiguous (asking about capabilities)\n"
                "'cuánto cuesta una biopsia?' → ambiguous (price question about service area)\n"
                "'precio de biopsia de apendice' → ambiguous (specific procedure price — always ambiguous, never off_topic)\n"
                "'cuánto cuesta el estudio X?' → ambiguous (any study/procedure price = ambiguous)\n"
                "'como se hace una pizza?' → off_topic\n"
                "'dime el resultado del partido de fútbol' → off_topic\n"
                "'quiero hablar con un humano' → needs_human"
                f"{questions_hint}"
            ),
        },
        {"role": "user", "content": question},
    ]
    try:
        raw = await call_chat(messages, max_tokens=150, temperature=0.2)
        parsed = extract_json_from_llm_response(raw)
        return parsed["intent"], parsed["reply"]
    except Exception as e:
        logger.warning("_triage_response failed: %s", e)
        area_clause = f" en {expertise_area}" if expertise_area else ""
        return "ambiguous", f"No encontré información específica sobre eso{area_clause}. Para más detalles, contactanos directamente."


async def generate_answer(
    context_chunks: list[dict],
    question: str,
    conversation_history: list[dict],
    expertise_area: str = "",
    channel: str = "telegram",
    images: list[dict] | None = None,
    from_web: bool = False,
    low_confidence: bool = False,
    max_tokens: int = 800,
    no_length_limit: bool = False,
    on_failover: Callable[[], Awaitable[None]] | None = None,
    doc_structure_summary: str | None = None,
) -> str:
    """
    Generate an answer using retrieved context + conversation history.
    When images is set, sends the images alongside the question (vision models).
    When from_web is True, adds web-source framing to the system prompt.
    When low_confidence is True, context came from a second-pass retrieval with
    lower similarity threshold — the LLM should note the approximate match.
    When no_length_limit is True, overrides the channel length guidance so the
    LLM lists all items in context without truncating (full catalog queries).
    """
    if not context_chunks and not images:
        return "No encontré información relevante en los documentos para responder tu pregunta."

    system_prompt = build_system_prompt(expertise_area, channel=channel, from_web=from_web,
                                        no_length_limit=no_length_limit,
                                        doc_structure_summary=doc_structure_summary)

    # Vision calls: cap context at 5 chunks. Reasoning models scale reasoning cost with
    # input size; 19 chunks + image exhausts the entire max_tokens budget on chain-of-thought.
    # Chunks are sorted by relevance (descending similarity), so the top 5 are the most useful.
    if images and len(context_chunks) > 5:
        context_chunks = context_chunks[:5]

    if context_chunks:
        context_text = "\n\n---\n\n".join([
            f"[Source: {c['source']}, Page {c['page']}]\n{c['content']}"
            for c in context_chunks
        ])
        confidence_note = "\n\nNOTA: Los siguientes documentos son coincidencias aproximadas (no exactas). Proporciona la información que encuentres y aclara que puede ser similar pero no idéntico a lo que pregunta el usuario.\n" if low_confidence else ""
        text_content = (
            f"<document_context>\n{context_text}\n</document_context>{confidence_note}\n\n"
            f"<user_question>\n{question}\n</user_question>"
        )
    else:
        # Image-only: no text context, but images are present
        text_content = f"<user_question>\n{question}\n</user_question>"

    # When user sent image(s), add explicit instruction so the LLM processes
    # the images instead of quoting policies like "envíe la imagen por WhatsApp"
    if images:
        count_text = f"{len(images)} imágenes" if len(images) > 1 else "una imagen"
        image_instruction = (
            f"\n\n[INSTRUCCIÓN IMPORTANTE: El usuario ya envió {count_text}. "
            "Analiza la(s) imagen(es) que recibiste y responde basándote en lo que ves. "
            "NUNCA le digas al usuario que envíe una imagen o que contacte por WhatsApp "
            "para enviar una imagen — YA la envió. Si la imagen contiene una orden "
            "médica o documento, extrae la información y responde con los precios "
            "que encuentres en el contexto. "
            "Si la imagen está PARCIALMENTE legible (puedes leer algunas partes pero no todas): "
            "1) Proporciona la información que SÍ puedes leer. "
            "2) Aclara explícitamente qué partes no se pudieron leer (ej: 'No se pudo leer el monto de X'). "
            "3) Sugiere enviar una imagen más clara solo para las partes que no se pudieron leer. "
            "NUNCA descartes toda la imagen si puedes leer algo — siempre extrae lo que puedas.]"
        )
        text_content += image_instruction

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(conversation_history[-6:])

    if images:
        # OpenAI vision format: content is a list with text + multiple image_url parts
        content: list[dict] = [{"type": "text", "text": text_content}]
        for img in images:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:{img['mime']};base64,{img['b64']}"},
            })
        messages.append({"role": "user", "content": content})
    else:
        messages.append({"role": "user", "content": text_content})

    # Reasoning-model fallbacks (e.g. mimo-v2.5) split max_tokens between
    # chain-of-thought and content. Vision analysis of complex documents needs
    # headroom so content isn't truncated mid-sentence.
    effective_max_tokens = max(max_tokens, 2000) if images else max_tokens

    vision_model = settings.llm_vision_model or None
    try:
        return await call_chat(
            messages,
            max_tokens=effective_max_tokens,
            temperature=0.1,
            channel=channel,
            model=vision_model if images else None,
            on_failover=on_failover if not images else None,
        )
    except RuntimeError:
        if not images:
            raise
        # Vision primary failed (e.g. rate-limit). Retry via the normal fallback
        # chain WITHOUT an explicit model= so fallback is allowed. Images are kept:
        # the fallback model (mimo-v2.5) is omnimodal and handles image payloads.
        logger.warning("generate_answer: vision call failed — retrying with fallback chain (images preserved)")
        return await call_chat(
            messages,
            max_tokens=effective_max_tokens,
            temperature=0.1,
            channel=channel,
        )


# ─── Conversation History ─────────────────────────────────────────────────────

async def get_history(
    db: AsyncSession,
    user_id: str,
    namespace: str,
    limit: int = 10,
) -> list[dict]:
    result = await db.execute(
        select(Conversation)
        .where(
            Conversation.user_id == user_id,
            Conversation.namespace == namespace,
        )
        .order_by(Conversation.created_at.desc())
        .limit(limit)
    )
    rows = list(reversed(result.scalars().all()))
    history = []
    for r in rows:
        if r.role == "user":
            try:
                content = sanitize_user_input(r.content)
            except ValueError:
                content = "[message removed]"
        elif r.role == "system":
            # Conversation summaries — pass through as-is for LLM context
            content = r.content
        elif r.role == "assistant" and CANARY_TOKEN in r.content:
            content = "[message redacted]"
        else:
            content = r.content
        history.append({"role": r.role, "content": content})
    return history


HISTORY_ROW_CAP = 50  # max rows per user per namespace (~25 turns)
SUMMARY_THRESHOLD = 30  # rows above which old history gets summarized (~15 turns)
SUMMARY_KEEP = 10       # rows of recent history to keep intact after summarization


async def _summarize_old_history(
    db: AsyncSession,
    user_id: str,
    namespace: str,
    tenant_id: int | None = None,
) -> None:
    """When conversation history exceeds SUMMARY_THRESHOLD rows, compact the
    oldest turns into a single system summary row using an LLM call.

    This keeps context within LLM token limits while preserving conversation
    continuity. The most recent SUMMARY_KEEP rows are left intact.
    Called from save_turn() after each turn is persisted.
    """
    from security import CANARY_TOKEN

    # Check row count
    count_result = await db.execute(
        text("SELECT COUNT(*) FROM conversations WHERE user_id = :uid AND namespace = :ns"),
        {"uid": user_id, "ns": namespace},
    )
    total = count_result.scalar_one()
    if total <= SUMMARY_THRESHOLD:
        return

    # Fetch oldest rows (all except the most recent SUMMARY_KEEP)
    old_rows_result = await db.execute(
        select(Conversation)
        .where(Conversation.user_id == user_id, Conversation.namespace == namespace)
        .order_by(Conversation.created_at)
        .limit(total - SUMMARY_KEEP)
    )
    old_rows = old_rows_result.scalars().all()
    if not old_rows:
        return

    # Build conversation text for summarization
    conv_lines = []
    for r in old_rows:
        role = "Usuario" if r.role == "user" else "Asistente"
        content = r.content[:200]  # truncate long messages
        conv_lines.append(f"{role}: {content}")

    conv_text = "\n".join(conv_lines)

    messages = [
        {
            "role": "system",
            "content": (
                "Resumí la siguiente conversación en 2-3 oraciones en español, "
                "manteniendo los temas clave discutidos, cualquier información "
                "pendiente o pregunta sin responder, y datos importantes mencionados "
                "(precios, horarios, nombres). No inventes información que no esté "
                "en la conversación. Respondé SOLO con el resumen, sin prefijo."
            ),
        },
        {"role": "user", "content": conv_text},
    ]

    try:
        summary = await call_chat(messages, max_tokens=200, temperature=0.0)
        summary = summary.strip()
    except Exception as e:
        logger.warning("_summarize_old_history: LLM call failed: %s — skipping", e)
        return

    # Redact canary token if present
    if CANARY_TOKEN in summary:
        summary = summary.replace(CANARY_TOKEN, "[REDACTED]")

    # Delete the old rows and insert summary row
    old_ids = [r.id for r in old_rows]
    id_list = ",".join(str(i) for i in old_ids)
    await db.execute(
        text(f"DELETE FROM conversations WHERE id IN ({id_list})"),
    )

    db.add(Conversation(
        user_id=user_id,
        namespace=namespace,
        role="system",
        content=f"[Resumen de conversación previa]: {summary}",
        channel="system",
        tenant_id=tenant_id,
    ))
    await db.commit()
    logger.info("_summarize_old_history: summarized %d old rows for user=%s ns=%s",
                len(old_rows), user_id, namespace)


async def save_turn(
    db: AsyncSession,
    user_id: str,
    namespace: str,
    user_msg: str,
    assistant_msg: str,
    channel: str = "telegram",
    tenant_id: int | None = None,
):
    db.add(Conversation(
        user_id=user_id,
        namespace=namespace,
        role="user",
        content=user_msg,
        channel=channel,
        tenant_id=tenant_id,
    ))
    db.add(Conversation(
        user_id=user_id,
        namespace=namespace,
        role="assistant",
        content=assistant_msg,
        channel=channel,
        tenant_id=tenant_id,
    ))
    await db.commit()

    # Trim old rows, keeping only the most recent HISTORY_ROW_CAP entries
    await db.execute(
        text("""
            DELETE FROM conversations
            WHERE user_id = :uid AND namespace = :ns
            AND id NOT IN (
                SELECT id FROM conversations
                WHERE user_id = :uid AND namespace = :ns
                ORDER BY created_at DESC
                LIMIT :cap
            )
        """),
        {"uid": user_id, "ns": namespace, "cap": HISTORY_ROW_CAP},
    )
    await db.commit()

    # Summarize old history when threshold is exceeded (best-effort)
    try:
        await _summarize_old_history(db, user_id, namespace, tenant_id)
    except Exception as e:
        logger.warning("save_turn: summarize failed for user=%s ns=%s: %s", user_id, namespace, e)


# ─── Web Search ──────────────────────────────────────────────────────────────────

# Ollama Cloud sends {"query": ..., "num_results": N} and returns
# {"results": [{"title": ..., "body": ..., "url": ...}, ...]}.
# Generic providers (Tavily, Brave) return different shapes — we normalize.
_OLLAMA_SEARCH_BODY = lambda q: json.dumps({"query": q, "num_results": 5}).encode()
_GENERIC_SEARCH_BODY = lambda q: json.dumps({"query": q}).encode()


# ─── Source citation footer ──────────────────────────────────────────────────

_SOURCE_EXTENSIONS = ('.md', '.pdf', '.txt', '.docx', '.doc', '.csv', '.xlsx', '.xls', '.pptx', '.ppt')


def _normalize_source_name(name: str) -> str:
    """Strip file extension and replace hyphens/underscores with spaces.

    Turns internal filenames like 'sp-diagnostico-histologico.md' into
    human-readable labels like 'Sp Diagnostico Histologico'.
    """
    for ext in _SOURCE_EXTENSIONS:
        if name.lower().endswith(ext):
            name = name[:-len(ext)]
            break
    name = name.replace('-', ' ').replace('_', ' ')
    name = name.strip()
    # Title-case each word (preserves existing capitals)
    name = ' '.join(word[0].upper() + word[1:] if word else '' for word in name.split())
    return name


def _build_source_footer(chunks: list[dict] | None, channel: str = "telegram") -> str:
    """Build a source citation footer from retrieved chunks.

    Collects document sources (with page numbers) and web URLs from chunks,
    deduplicates by source name, and formats per-channel conventions.
    Source filenames are normalized to human-readable names (stripped of
    extensions, hyphens replaced with spaces, title-cased).

    Returns empty string if no attribution info is available.
    """
    if not chunks:
        return ""

    doc_sources: dict[str, set[int]] = {}  # source_name → {pages}
    web_urls: list[str] = []
    seen_urls: set[str] = set()

    for c in chunks:
        src = c.get("source", "")
        if not src or src == "__faq__":
            continue
        if src.startswith("http"):
            if src not in seen_urls:
                web_urls.append(src)
                seen_urls.add(src)
        else:
            page = c.get("page")
            pages = doc_sources.setdefault(src, set())
            if page and int(page) > 0:
                pages.add(int(page))

    if not doc_sources and not web_urls:
        return ""

    parts: list[str] = []
    for name, pages in doc_sources.items():
        display_name = _normalize_source_name(name)
        sorted_pages = sorted(pages)
        if sorted_pages:
            parts.append(f"{display_name} p.{','.join(str(p) for p in sorted_pages)}")
        else:
            parts.append(display_name)

    parts.extend(web_urls)
    joined = ", ".join(parts)
    if channel == "whatsapp":
        return f"\n\n📎 Fuentes: {joined}"
    return f"\n\n📎 _Fuentes: {joined}_"


async def _web_search(question: str) -> list[dict]:
    """
    Call the configured web search endpoint and return context chunks.
    Tries Ollama format first; if 404/405, retries with generic format.
    Returns [] on any error (best-effort, never blocks reply).
    Each chunk: {"content": str, "source": str, "page": 0, "similarity": 0.4}
    """
    web_search_url = get_setting("web_search_url", settings.web_search_url)
    if not web_search_url:
        return []

    api_key = get_setting("llm_api_key", settings.effective_llm_api_key)
    headers = {
        "Content-Type": "application/json",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    try:
        # Try Ollama-style endpoint (with num_results param)
        response = await http_client.post(
            web_search_url,
            content=_OLLAMA_SEARCH_BODY(question),
            headers=headers,
        )

        # If endpoint rejects Ollama format (404/405), try generic format
        if response.status_code in (404, 405):
            logger.info("web_search_ollama_format_rejected url=%s status=%d, retrying generic",
                        web_search_url, response.status_code)
            response = await http_client.post(
                web_search_url,
                content=_GENERIC_SEARCH_BODY(question),
                headers=headers,
            )

        response.raise_for_status()
        data = response.json()

        # Normalize: Ollama returns {"results": [...]} with "body" field
        # Generic providers may return {"results": [...]} with "content"/"text"/"snippet"
        raw_results = data.get("results", data if isinstance(data, list) else [])
        chunks = []
        for r in raw_results:
            # Extract text content from various field names
            content = (
                r.get("body") or r.get("content") or r.get("text")
                or r.get("snippet") or r.get("description") or ""
            )
            if not content or len(content.strip()) < 50:
                continue
            url = r.get("url") or r.get("link") or r.get("source") or ""
            # Include URL in content for attribution
            if url:
                content = f"{content}\nFuente: {url}"
            chunks.append({
                "content": content.strip(),
                "source": url or "web",
                "page": 0,
                "similarity": 0.4,  # lower than doc matches — web sources are less authoritative
            })

        return chunks

    except Exception as e:
        logger.warning("web_search_failed url=%s error=%s", web_search_url, e)
        return []


# ─── Tool use definitions ─────────────────────────────────────────────────────

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_documents",
            "description": (
                "Search the company's indexed documents (catalogs, manuals, policies, FAQs) "
                "for information about the company's own products, services, prices, rules, or procedures."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Standalone search query."}
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": (
                "Search the web for current information not in the company documents: "
                "external facts, regulations, competitor info, recent events."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Web search query."}
                },
                "required": ["query"],
            },
        },
    },
]

# ─── Index status tracking ────────────────────────────────────────────────────
# (namespace, source) → "procesando" | "activo". Set before index_chunks, updated
# after success. Entries auto-expire after 5 min (lazy TTL check on read).
# PR2 portal UI reads this for the ◐ procesando badge.

_index_status: dict[tuple[str, str], tuple[str, float]] = {}  # key → (status, monotonic_ts)
_INDEX_STATUS_TTL = 300  # 5 minutes


def set_index_status(namespace: str, source: str, status: str) -> None:
    """Set index status: 'procesando' or 'activo'."""
    _index_status[(namespace, source)] = (status, time.monotonic())


def get_index_status(namespace: str, source: str) -> str | None:
    """Return 'procesando' or 'activo', or None if not tracked / expired."""
    entry = _index_status.get((namespace, source))
    if entry is None:
        return None
    status, ts = entry
    if time.monotonic() - ts > _INDEX_STATUS_TTL:
        del _index_status[(namespace, source)]
        return None
    return status


# ─── Tool result cache ────────────────────────────────────────────────────────
# Keyed by "namespace:tool_name:sha256(query)[:16]". Value: (result_str, chunks, monotonic_ts).
# LRU cap: 1000 entries (FIFO eviction on overflow). TTL: 300s (lazy eviction on read).
# Flushed on document upsert via flush_tool_cache(namespace).

_tool_cache: OrderedDict[str, tuple[str, list[dict], float]] = OrderedDict()
_TOOL_CACHE_MAX = 1000


def _cache_key(namespace: str, tool_name: str, query: str) -> str:
    h = hashlib.sha256(query.encode()).hexdigest()[:16]
    return f"{namespace}:{tool_name}:{h}"


def _get_cached(namespace: str, tool_name: str, query: str) -> tuple[str, list[dict]] | None:
    key = _cache_key(namespace, tool_name, query)
    if key in _tool_cache:
        result, chunks, ts = _tool_cache[key]
        if time.monotonic() - ts < 300:
            return result, chunks
        del _tool_cache[key]
    return None


def _set_cached(namespace: str, tool_name: str, query: str, result: str, chunks: list[dict]) -> None:
    key = _cache_key(namespace, tool_name, query)
    if len(_tool_cache) >= _TOOL_CACHE_MAX:
        _tool_cache.popitem(last=False)  # evict oldest entry (FIFO-LRU)
    _tool_cache[key] = (result, chunks, time.monotonic())


def flush_tool_cache(namespace: str) -> None:
    """Remove all cached tool results for a namespace. Call after document upsert."""
    prefix = f"{namespace}:"
    to_del = [k for k in _tool_cache if k.startswith(prefix)]
    for k in to_del:
        del _tool_cache[k]
    if to_del:
        logger.debug("tool_cache flushed: %s (%d entries)", namespace, len(to_del))


# ─── Tool dispatch helpers ────────────────────────────────────────────────────

async def _tool_search_documents(
    query: str, namespace: str, expertise_area: str
) -> tuple[str, list[dict]]:
    """Search indexed documents for a tool dispatch call.

    Uses a fresh DB session (never reentrant with outer rag_query session).
    HyDE runs inside the fresh session; cache is keyed on reformulated query (pre-HyDE).
    Returns (formatted_string, chunk_list). Empty string + [] when no matches found.
    """
    cached = _get_cached(namespace, "search_documents", query)
    if cached is not None:
        logger.debug("tool_cache hit: %s:search_documents", namespace)
        return cached

    async with tenant_session(namespace) as db:
        hyde_q = await _hyde_query(query, expertise_area)
        search_q = hyde_q if hyde_q else query
        chunks = await retrieve_context(db, search_q, namespace)

    if not chunks:
        _set_cached(namespace, "search_documents", query, "", [])
        return "", []

    formatted = "\n\n".join(f"[Doc {i + 1}]: {c['content']}" for i, c in enumerate(chunks))
    _set_cached(namespace, "search_documents", query, formatted, chunks)
    logger.debug("tool_call: search_documents → %d chars, %d chunks", len(formatted), len(chunks))
    return formatted, chunks


async def _tool_search_web(query: str, tenant: "Tenant") -> tuple[str, list[dict]]:
    """Search the web for a tool dispatch call.

    Returns (formatted_string, web_chunks) where web_chunks are the
    raw search results suitable for source attribution.
    """
    if not get_setting("web_search_url", settings.web_search_url):
        return "", []

    namespace = tenant.slug
    cached = _get_cached(namespace, "search_web", query)
    if cached is not None:
        logger.debug("tool_cache hit: %s:search_web", namespace)
        result, chunks = cached
        return result, chunks

    web_chunks = await _web_search(query)
    result = "\n\n".join(f"[Web]: {c['content']}" for c in web_chunks) if web_chunks else ""
    _set_cached(namespace, "search_web", query, result, web_chunks)
    logger.debug("tool_call: search_web → %d chars", len(result))
    return result, web_chunks


async def _dispatch_tool(tool_call: dict, namespace: str, tenant: "Tenant") -> str:
    """Execute a single tool call and return its string result."""
    name = tool_call["function"]["name"]
    inp = json.loads(tool_call["function"]["arguments"])
    if name == "search_documents":
        result, _ = await _tool_search_documents(inp["query"], namespace, tenant.expertise_area or "")
        return result or "No relevant documents found."
    if name == "search_web":
        result, _ = await _tool_search_web(inp["query"], tenant)
        return result or "No web results found."
    logger.warning("_dispatch_tool: unknown tool=%s", name)
    return "Unknown tool."


async def _log_unanswered(
    db: AsyncSession,
    namespace: str,
    question: str,
    user_id: str,
    intent: str,
    tenant_id: int | None = None,
) -> None:
    try:
        db.add(UnansweredQuery(
            tenant_id=tenant_id,
            namespace=namespace,
            question=question,
            user_id=user_id,
            intent_category=intent,
        ))
        await db.commit()
    except Exception as e:
        logger.warning("Failed to log UnansweredQuery: %s", e)


async def _classify_intent(question: str, last_bot_turn: str | None = None, expertise_area: str = "") -> str:
    """
    Lightweight pre-RAG intent classification. Called after the tool-use block in rag_query().
    Returns one of: search_docs | needs_human | price_catalog
    Falls back to "search_docs" on any LLM or parse failure (safe default).
    When expertise_area is provided, counter-examples use domain-specific terminology.
    """
    # Derive a singular noun for counter-examples from expertise_area
    area_singular = ""
    if expertise_area:
        # Take the first noun-like word (e.g. "diagnóstico histológico" → "diagnóstico")
        words = expertise_area.strip().split()
        area_singular = words[0].lower() if words else expertise_area.lower()

    context_hint = (
        f"\nPrevious bot reply (context only): {last_bot_turn[:120]}"
        if last_bot_turn else ""
    )
    # E3: Dynamic counter-examples based on expertise_area
    price_counter = (
        f"  'cuánto cuesta {area_singular}' → search_docs\n"
        f"  'precio del {area_singular}' → search_docs\n"
        "  'dame todos los precios' → price_catalog"
        if area_singular
        else "  'cuánto cuesta X' → search_docs\n"
        "  'precio del estudio X' → search_docs\n"
        "  'dame todos los precios' → price_catalog"
    )
    messages = [
        {"role": "system", "content": (
            "Classify the user's intent. Reply ONLY with JSON: "
            '{"intent": "<search_docs|needs_human|price_catalog>"}\n'
            "Definitions:\n"
            "- search_docs: user wants information from documents (DEFAULT — when in doubt)\n"
            "- needs_human: user EXPLICITLY wants to speak with a real person right now\n"
            "- price_catalog: user wants the COMPLETE price list for all services\n"
            "Counter-examples for price_catalog:\n"
            f"{price_counter}\n"
            "Counter-examples for needs_human:\n"
            "  '¿cómo puedo contactar?' → search_docs\n"
            "  'necesito contactar para saber mis resultados' → search_docs\n"
            "  'quiero hablar con un humano' → needs_human\n"
            "  'necesito hablar con una persona' → needs_human"
            f"{context_hint}"
        )},
        {"role": "user", "content": question},
    ]
    intent_model = get_setting("llm_intent_model", settings.llm_intent_model) or None
    try:
        raw = await call_chat(messages, max_tokens=80, temperature=0.0, model=intent_model)
        parsed = extract_json_from_llm_response(raw)
        intent = parsed.get("intent", "search_docs")
        if intent not in {"search_docs", "needs_human", "price_catalog"}:
            return "search_docs"
        return intent
    except Exception as e:
        logger.warning("classify_intent failed: %s — fallback search_docs", e)
        return "search_docs"


# ─── E1: Faithfulness self-check ──────────────────────────────────────────────
# When low_confidence=True AND top similarity < 0.85, run a second LLM call
# to verify the answer is grounded in the provided context. If not, append a caveat.

_FAITHFULNESS_CEILING = 0.85


async def _faithfulness_check(
    question: str,
    answer: str,
    context: list[dict],
    channel: str = "telegram",
) -> str:
    """Verify that the answer is grounded in the provided context.

    Returns the original answer if faithful, or the answer with a caveat appended
    if the LLM identifies unsupported claims.
    """
    context_text = "\n\n".join(
        f"[Source: {c['source']}, Page {c.get('page', 0)}]\n{c['content']}"
        for c in context[:5]  # Limit context for the check call
    )
    messages = [
        {"role": "system", "content": (
            "You are a fact-checker. Given a question, an answer, and source context, "
            "verify that the answer is supported by the context. "
            "If the answer contains claims NOT supported by the context, respond with a JSON object: "
            '{"faithful": false, "unsupported_claims": "description of unsupported claims"}'
            "\nIf the answer is fully supported, respond: "
            '{"faithful": true}'
            "\n\nRespond ONLY with the JSON object."
        )},
        {"role": "user", "content": f"Question: {question}\n\nAnswer: {answer}\n\nContext:\n{context_text}"},
    ]
    try:
        result = await call_chat(messages, max_tokens=200, temperature=0.0)
        parsed = extract_json_from_llm_response(result)
        if parsed.get("faithful") is False:
            unsupported = parsed.get("unsupported_claims", "")
            caveat = f"\n\n⚠️ Aclaración: algunas afirmaciones pueden no estar completamente verificadas en los documentos disponibles. {unsupported}"
            return answer + caveat
    except Exception as e:
        logger.warning("faithfulness_check failed: %s", e)
    return answer


async def rag_query(
    db: AsyncSession,
    question: str,
    namespace: str,
    user_id: str,
    expertise_area: str = "",
    language_code: str | None = None,
    tenant_id: int | None = None,
    channel: str = "telegram",
    images: list[dict] | None = None,
    tenant: "Tenant | None" = None,
) -> tuple[str, list[dict], str | None]:
    """
    Full RAG pipeline: retrieve context → generate answer → save history.
    Returns (answer, retrieved_chunks, intent | None).
    intent is None when answered from docs; otherwise the triage classification.
    When images is set, the images are passed to generate_answer() for vision models.
    When tenant is provided and web_search_enabled, falls back to web search
    when no context is found in the knowledge base.
    """
    # Pre-RAG: local greeting classifier — skip LLM for obvious greetings
    if _GREETING_PATTERN.match(question.strip()):
        # Tenant-adaptive greeting: use example_questions if available
        _eq = getattr(tenant, "example_questions", None) if tenant else None
        if _eq and len(_eq) > 0:
            # Suggest up to 2 example questions
            suggestions = _eq[:2]
            suggestion_text = " Por ejemplo:\n" + "\n".join(f"• {s}" for s in suggestions)
            answer = f"¡Hola! ¿En qué puedo ayudarte?{suggestion_text}"
        elif expertise_area:
            answer = f"¡Hola! ¿En qué puedo ayudarte con {expertise_area}?"
        else:
            answer = "¡Hola! ¿En qué puedo ayudarte?"
        await save_turn(db, user_id, namespace, question, answer, channel=channel, tenant_id=tenant_id)
        return answer, [], "greeting"

    # Vision guard: if user sent an image but no vision model is configured,
    # skip the LLM call entirely — sending image payloads to text-only models
    # produces opaque 404 errors from the provider.
    if images and not get_setting("llm_vision_model", settings.llm_vision_model):
        logger.info("vision_guard ns=%s user=%s — no vision model configured", namespace, user_id)
        answer = "No puedo procesar imágenes en este momento. Por favor, envía tu consulta por texto."
        img_label = "📷 [varias imágenes]" if len(images) > 1 else "📷 [imagen]"
        await save_turn(db, user_id, namespace, question or img_label, answer, channel=channel, tenant_id=tenant_id)
        return answer, [], "no_vision_model"

    # Query reformulation: resolve pronouns/references using conversation history
    # Use reformulated query for vector search, original question for LLM answer generation
    history = await get_history(db, user_id, namespace)
    search_query = await _reformulate_query(question, history)
    if search_query != question:
        logger.info("reformulate ns=%s original=%r → reformulated=%r",
                     namespace, question[:60], search_query[:60])

    # E5: Extract doc_structure_summary from tenant for system prompt enrichment
    _doc_summary = getattr(tenant, "doc_structure_summary", None) if tenant else None

    # Build failover alert callback (one per rag_query call, shared by all generate_answer calls)
    _on_failover: Callable[[], Awaitable[None]] | None = None
    if tenant:
        _on_failover = lambda: _alert_llm_failover(tenant)  # type: ignore[misc]

    # ── TOOL PATH ────────────────────────────────────────────────────────────────
    # When 2+ tools are available and the LLM provider supports tool_use, let the
    # LLM choose which tools to call (Round 1), dispatch them in parallel, then
    # synthesize from all results (Round 2). Falls through to sequential pipeline
    # on failure, provider incompatibility, or all-empty tool results.
    _web_available = (
        (tenant.web_search_enabled if tenant else False)
        and bool(get_setting("web_search_url", settings.web_search_url))
    )
    _available_tools = TOOLS if _web_available else [TOOLS[0]]

    if is_tool_use_available() and not images and len(_available_tools) >= 2:
        try:
            _system_prompt = build_system_prompt(expertise_area, channel=channel,
                                                   doc_structure_summary=_doc_summary)
            r1_content, tool_calls = await call_chat_with_tools(
                messages=[*history[-6:], {"role": "user", "content": question}],
                tools=_available_tools,
                system=_system_prompt,
                tool_choice="auto",
            )

            if r1_content:
                # LLM answered directly without tools (greeting, off-topic etc.)
                r1_content = validate_output(r1_content, user_id=user_id)
                if CANARY_TOKEN in r1_content:
                    r1_content = r1_content.replace(CANARY_TOKEN, "[REDACTED]")
                    logger.warning("canary_redacted user_id=%s tool_direct", user_id)
                await save_turn(db, user_id, namespace, question, r1_content, channel=channel, tenant_id=tenant_id)
                return r1_content, [], "direct"

            if tool_calls:
                # Parallel dispatch — partial failure handled via return_exceptions
                raw_results = await asyncio.gather(*[
                    _dispatch_tool(tc, namespace, tenant) for tc in tool_calls
                ], return_exceptions=True)

                tool_results: list[str] = []
                for tc, r in zip(tool_calls, raw_results):
                    if isinstance(r, Exception):
                        logger.warning(
                            "tool_dispatch_failed tool=%s error=%s",
                            tc["function"]["name"], r,
                        )
                        tool_results.append("")
                    else:
                        tool_results.append(r)

                if not all(r == "" for r in tool_results):
                    # Collect chunks for source attribution (cache hit — no extra DB call)
                    _attribution_chunks: list[dict] = []
                    for tc in tool_calls:
                        fname = tc["function"]["name"]
                        q = json.loads(tc["function"]["arguments"])["query"]
                        if fname == "search_documents":
                            _, chunks = await _tool_search_documents(q, namespace, expertise_area or "")
                            _attribution_chunks.extend(chunks)
                        elif fname == "search_web":
                            _, chunks = await _tool_search_web(q, tenant)
                            _attribution_chunks.extend(chunks)

                    synthesis_messages = [
                        {"role": "system", "content": _system_prompt},
                        *history[-6:],
                        {"role": "user", "content": question},
                        {"role": "assistant", "tool_calls": tool_calls},
                        *[
                            {"role": "tool", "tool_call_id": tc["id"], "content": result}
                            for tc, result in zip(tool_calls, tool_results)
                        ],
                    ]
                    answer = await call_chat(messages=synthesis_messages, max_tokens=800, channel=channel)
                    answer = validate_output(answer, user_id=user_id)
                    if CANARY_TOKEN in answer:
                        answer = answer.replace(CANARY_TOKEN, "[REDACTED]")
                        logger.warning("canary_redacted user_id=%s tool_path", user_id)
                    await save_turn(db, user_id, namespace, question, answer, channel=channel, tenant_id=tenant_id)
                    return answer, _attribution_chunks, "tool_use"

                # All tool results empty — fall through to sequential pipeline
                logger.warning(
                    "tool_path_all_empty ns=%s q=%r — falling back to sequential",
                    namespace, question[:60],
                )

        except ToolUseNotSupportedError:
            logger.warning("tool_use_not_supported ns=%s — falling back to sequential", namespace)
        except Exception as e:
            logger.warning("tool_path_error ns=%s error=%s — falling back to sequential", namespace, e)
    # ── END TOOL PATH ─────────────────────────────────────────────────────────

    # ── INTENT ROUTER ─────────────────────────────────────────────────────────
    # LLM-based classifier replaces ESCALATION_PATTERN + _PRICE_INTENT_RE regex gates.
    # Runs only when tool-use falls through (tool-use success returns early above).
    last_bot_turn = next(
        (m["content"] for m in reversed(history) if m.get("role") == "assistant"), None
    )
    intent = await _classify_intent(question, last_bot_turn=last_bot_turn, expertise_area=expertise_area)
    logger.info("classify_intent ns=%s intent=%s q=%r", namespace, intent, question[:60])

    if intent == "needs_human":
        area_clause = f" Mi área de expertise: {expertise_area}." if expertise_area else ""
        answer = f"Entiendo que quieres hablar con alguien.{area_clause} Contactamos directamente."
        await save_turn(db, user_id, namespace, question, answer, channel=channel, tenant_id=tenant_id)
        await _log_unanswered(db, namespace, question, user_id, "needs_human", tenant_id)
        logger.info("unanswered_escalation ns=%s source=intent_router q=%r", namespace, question[:60])
        return answer, [], "needs_human"

    if intent == "price_catalog":
        full_chunks = await retrieve_full_catalog(db, namespace)
        if full_chunks:
            logger.info("catalog_llm ns=%s q=%r items=%d", namespace, question[:60], len(full_chunks))
            answer = await _format_catalog_with_llm(full_chunks, channel=channel, on_failover=_on_failover)
            answer = validate_output(answer, user_id=user_id)
            if CANARY_TOKEN in answer:
                answer = answer.replace(CANARY_TOKEN, "[REDACTED]")
                logger.warning("canary_redacted user_id=%s catalog", user_id)
            await save_turn(db, user_id, namespace, question, answer, channel=channel, tenant_id=tenant_id)
            return answer, full_chunks, None
        logger.info("catalog_llm ns=%s — no section chunks, falling through", namespace)
    # ── END INTENT ROUTER ─────────────────────────────────────────────────────

    # HyDE: embed a hypothetical answer instead of the question to bridge vocabulary gap
    broad_query = search_query  # pre-HyDE query: broader, used as retrieval supplement
    if get_setting("hyde_enabled", "on") == "on":
        hyde_result = await _hyde_query(search_query, expertise_area)
        if hyde_result:
            logger.info("hyde ns=%s q=%r → hypothetical=%r",
                        namespace, search_query[:60], hyde_result[:60])
            search_query = hyde_result

    context = await retrieve_context(db, search_query, namespace)

    # Dual retrieval: when HyDE specialized the query, also search with the pre-HyDE
    # (broad) query to recover chunks from different catalog sections that HyDE crowds out.
    if search_query != broad_query:
        context_broad = await retrieve_context(db, broad_query, namespace)
        seen = {c["content"] for c in context}
        new_from_broad = [c for c in context_broad if c["content"] not in seen]
        if new_from_broad:
            context = context + new_from_broad
            logger.info(
                "dual_retrieve ns=%s broad_q=%r added=%d total=%d",
                namespace, broad_query[:60], len(new_from_broad), len(context),
            )

    # Catalog overview: 1 representative chunk per section (cheap DB query, no embedding).
    # Collected now, merged into context after the similarity filter so catalog sections
    # don't rescue off-topic queries from triage (which requires context=[] to trigger).
    overview_chunks = await retrieve_catalog_overview(db, namespace)

    logger.info(
        "retrieve ns=%s q=%r search_q=%r top_scores=%s",
        namespace,
        question[:60],
        search_query[:60] if search_query != question else "(same)",
        [(round(c["similarity"], 3), c["source"], c["content"][:40]) for c in context[:3]],
    )
    raw_results = context  # keep unfiltered for low-confidence fallback
    context = [c for c in context if c["similarity"] >= MIN_SIMILARITY]
    is_low_confidence = False

    # Low-confidence fallback: when normal threshold filters everything out but
    # raw results exist above LOW_MIN_SIMILARITY, use them as approximate matches.
    # The COINCIDENCIAS PARCIALES prompt section guides the LLM to handle these
    # with appropriate uncertainty ("puede ser el mismo estudio", "contactanos para confirmar").
    if not context:
        low_context = [c for c in raw_results if c["similarity"] >= LOW_MIN_SIMILARITY]
        if low_context:
            logger.info(
                "low_confidence_fallback ns=%s q=%r top_sim=%.3f — using approximate matches",
                namespace, question[:60], low_context[0]["similarity"],
            )
            context = low_context
            is_low_confidence = True

    if not context:
        # Vision-augmented retrieval: when user sent image(s) but no text context
        # was found, use the vision model to extract key terms from the image,
        # then retry the vector search with those terms. This handles the common
        # case where the user sends a photo of a medical order with no caption —
        # the generic default question won't match any documents, but the actual
        # study names in the image (e.g. "Biopsia de apéndice cecal") will.
        if images:
            vision_legibility, vision_query = await _extract_search_terms_from_images(images)
            if vision_legibility == "illegible":
                answer = _illegible_fallback_msg(images)
                logger.info("vision_illegible ns=%s user=%s images=%d", namespace, user_id, len(images))
                img_label = "📷 [varias imágenes]" if len(images) > 1 else "📷 [imagen]"
                await save_turn(db, user_id, namespace, question or img_label, answer, channel=channel, tenant_id=tenant_id)
                return answer, [], None
            if vision_query:
                # Sanitize LLM-extracted query (same guard as user input) and limit length.
                # LLM output can contain injection patterns; if sanitization rejects it,
                # fall back to the raw query truncated rather than crashing the pipeline.
                try:
                    vision_query = sanitize_user_input(vision_query)
                except ValueError:
                    logger.warning("vision_query_sanitization_failed ns=%s user=%s", namespace, user_id)
                    vision_query = vision_query[:200]
                else:
                    vision_query = vision_query[:200]
            if vision_query and vision_query.strip().lower() != search_query.strip().lower():
                logger.info(
                    "vision_augmented_retrieval ns=%s user=%s original_q=%r vision_q=%r",
                    namespace, user_id, search_query[:60], vision_query[:60],
                )
                vision_context = await retrieve_context(db, vision_query, namespace)
                vision_raw = vision_context
                vision_context = [c for c in vision_context if c["similarity"] >= MIN_SIMILARITY]
                if not vision_context:
                    # Try low-confidence fallback with vision-extracted terms
                    vision_context = [c for c in vision_raw if c["similarity"] >= LOW_MIN_SIMILARITY]
                    if vision_context:
                        is_low_confidence = True
                if vision_context:
                    context = vision_context
                    raw_results = vision_raw
                    search_query = vision_query
                    logger.info(
                        "vision_retrieval_hit ns=%s q=%r top_scores=%s",
                        namespace, vision_query[:60],
                        [(round(c["similarity"], 3), c["source"], c["content"][:40]) for c in context[:3]],
                    )

    # Merge catalog overview sections into context when we have some similarity results.
    # Overview gives 1 representative chunk per section for listing/broad queries.
    # Not merged when context is empty — off-topic queries intentionally need
    # context=[] so triage fires correctly rather than generating from catalog headers.
    if context and overview_chunks:
        seen = {c["content"] for c in context}
        catalog_new = [c for c in overview_chunks if c["content"] not in seen]
        if catalog_new:
            context = context + catalog_new
            logger.info(
                "catalog_sections_merged ns=%s added=%d total=%d",
                namespace, len(catalog_new), len(context),
            )

    # E7: Always-include policy chunks — policy_statement and section_header chunks
    # contain requirements, conditions, and structural context that semantic search
    # often misses because they score low on similarity to price/procedure queries.
    # Only merged when context is non-empty — off-topic queries should still triage.
    if context:
        policy_chunks = await retrieve_policy_chunks(db, namespace)
        if policy_chunks:
            seen = {c["content"] for c in context}
            policy_new = [c for c in policy_chunks if c["content"] not in seen]
            if policy_new:
                context = context + policy_new
                logger.info(
                    "policy_chunks_merged ns=%s q=%r added=%d total=%d",
                    namespace, question[:60], len(policy_new), len(context),
                )

    # E9: Cross-section retrieval — when price_row chunks are in context, fetch ALL
    # chunks from the same section. This catches procedure-specific requirements
    # (e.g. "Biopsia Extemporánea: traer cita, pago anticipado, sin formol...") that
    # are in the same section but have low semantic similarity to the price query.
    if context:
        price_rows = [c for c in context if c.get("chunk_type") == "price_row"]
        if price_rows:
            section_siblings = await retrieve_section_siblings(db, namespace, price_rows)
            if section_siblings:
                seen = {c["content"] for c in context}
                sibling_new = [c for c in section_siblings if c["content"] not in seen]
                if sibling_new:
                    context = context + sibling_new
                    logger.info(
                        "section_siblings_e9 ns=%s q=%r price_rows=%d added=%d total=%d",
                        namespace, question[:60], len(price_rows), len(sibling_new), len(context),
                    )

    # Context cap: after all merges (retrieval, catalog overview, policy, section siblings),
    # sort by similarity descending and truncate to MAX_CONTEXT_CHUNKS to prevent LLM context
    # window blowup. Semantic matches (similarity 0.7+) stay; policy/sibling chunks (0.5) are
    # kept only when there's room.
    if len(context) > MAX_CONTEXT_CHUNKS:
        context.sort(key=lambda c: c.get("similarity", 0), reverse=True)
        dropped = len(context) - MAX_CONTEXT_CHUNKS
        context = context[:MAX_CONTEXT_CHUNKS]
        logger.info(
            "context_capped ns=%s q=%r dropped=%d kept=%d",
            namespace, question[:60], dropped, len(context),
        )

    # If vision-augmented retrieval found context, fall through to the main
    # generate_answer path below. Otherwise, handle the no-context cases.
    if not context:
        # Image-only path (after vision-augmented retrieval also failed to find context):
        # process the images through the vision model without document context.
        if images:
            answer = await generate_answer(
                [], question, history, expertise_area,
                channel=channel, images=images,
                on_failover=_on_failover,
                doc_structure_summary=_doc_summary,
            )
            answer = validate_output(answer, user_id=user_id)
            if CANARY_TOKEN in answer:
                answer = answer.replace(CANARY_TOKEN, "[REDACTED]")
                logger.warning("canary_redacted user_id=%s image_only", user_id)
            img_label = "📷 [varias imágenes]" if len(images) > 1 else "📷 [imagen]"
            await save_turn(db, user_id, namespace, question or img_label, answer, channel=channel, tenant_id=tenant_id)
            return answer, [], None

        # Web search fallback: if tenant has web search enabled and URL is configured,
        # try web search before falling back to triage.
        web_search_enabled = tenant.web_search_enabled if tenant else False
        if web_search_enabled and get_setting("web_search_url", settings.web_search_url):
            web_results = await _web_search(question)
            if web_results:
                answer = await generate_answer(
                    web_results, question, history, expertise_area,
                    channel=channel, images=images,
                    from_web=True,
                    on_failover=_on_failover,
                    doc_structure_summary=_doc_summary,
                )
                answer = validate_output(answer, user_id=user_id)
                # Redact canary token at write time
                if CANARY_TOKEN in answer:
                    answer = answer.replace(CANARY_TOKEN, "[REDACTED]")
                    logger.warning("canary_redacted user_id=%s web_search", user_id)
                await save_turn(db, user_id, namespace, question, answer, channel=channel, tenant_id=tenant_id)
                return answer, web_results, "web_search"

        # No web search or no results — fall back to triage
        example_questions = tenant.example_questions if tenant else None
        intent, answer = await _triage_response(question, expertise_area, language_code, example_questions)
        answer = validate_output(answer, user_id=user_id)
        # Redact canary token at write time — prevents exfiltration via history
        if CANARY_TOKEN in answer:
            answer = answer.replace(CANARY_TOKEN, "[REDACTED]")
            logger.warning("canary_redacted user_id=%s intent=%s", user_id, intent)
        await save_turn(db, user_id, namespace, question, answer, channel=channel)
        await _log_unanswered(db, namespace, question, user_id, intent, tenant_id)
        return answer, [], intent

    answer = await generate_answer(
        context, question, history, expertise_area,
        channel=channel, images=images,
        low_confidence=is_low_confidence,
        on_failover=_on_failover,
        doc_structure_summary=_doc_summary,
    )

    # E1: Faithfulness self-check — only when low confidence AND top similarity below ceiling
    if is_low_confidence and context and context[0].get("similarity", 1.0) < _FAITHFULNESS_CEILING:
        answer = await _faithfulness_check(question, answer, context, channel=channel)

    answer = validate_output(answer, user_id=user_id)
    # Redact canary token at write time — prevents exfiltration via history
    if CANARY_TOKEN in answer:
        answer = answer.replace(CANARY_TOKEN, "[REDACTED]")
        logger.warning("canary_redacted user_id=%s context_answer", user_id)
    await save_turn(db, user_id, namespace, question, answer, channel=channel)
    return answer, context, None
