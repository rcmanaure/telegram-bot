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

import httpx
from services.prompts import build_system_prompt, ESCALATION_PATTERN, _GREETING_PATTERN
from services.stt import transcribe_voice
from security import CANARY_TOKEN, scan_chunk_for_injection, sanitize_user_input, validate_output

logger = logging.getLogger(__name__)
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession
from db import AsyncSessionLocal, DocumentChunk, Conversation, UnansweredQuery, Tenant
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
                "escribí en 1-2 oraciones un contexto breve que sitúe este fragmento dentro "
                "del documento. No repitas el contenido del fragmento. "
                "Respondé solo con el contexto, nada más.\n\n"
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
    return [
        {
            "content": row.content,
            "source": row.source,
            "page": row.page,
            "similarity": round(float(row.similarity), 3),
        }
        for row in rows
    ]


async def retrieve_catalog_overview(
    db: AsyncSession,
    namespace: str,
) -> list[dict]:
    """Return one representative chunk per catalog section for broad overview queries.

    Similarity search can't handle "what categories do you have" queries — it always
    concentrates top-k results in the sections that happen to match query terms best,
    starving all other sections. This function uses DISTINCT ON section header to
    guarantee exactly one representative chunk per section regardless of similarity.

    Only returns chunks whose content starts with a markdown section header (## ...),
    which is the format _split_markdown_tables produces for table-row chunks.
    Falls back gracefully to [] if the namespace has no such structured chunks.
    """
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
    return [
        {
            "content": row.content,
            "source": row.source,
            "page": row.page,
            "similarity": 0.5,
        }
        for row in rows
    ]


async def retrieve_full_catalog(
    db: AsyncSession,
    namespace: str,
) -> list[dict]:
    """Return ALL procedure chunks ordered by section for full price-list queries.

    Unlike retrieve_catalog_overview (1 per section), this returns every procedure
    in the catalog. Used when the user asks for prices across all categories —
    the LLM needs the complete dataset to enumerate every item.
    """
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
    return [
        {
            "content": row.content,
            "source": row.source,
            "page": row.page,
            "similarity": 0.5,
        }
        for row in rows
    ]


_CATALOG_SECTION_EMOJI: dict[str, str] = {
    "GINECOLÓGICO": "🌸",
    "GLÁNDULA MAMARIA": "🎀",
    "SISTEMA UROLÓGICO Y GENITAL MASCULINO": "🔵",
    "SISTEMA RESPIRATORIO": "🫁",
    "SISTEMA DIGESTIVO": "🍽️",
    "SISTEMA ENDOCRINO": "🦋",
    "SISTEMA CARDIO-CIRCULATORIO": "❤️",
    "SISTEMA OSTEO-MUSCULAR Y PARTES BLANDAS": "🦴",
    "SISTEMA NERVIOSO CENTRAL Y PERIFÉRICO": "🧠",
    "SISTEMA OCULAR": "👁️",
    "SISTEMA HEMATOPOYÉTICO Y GANGLIONAR LINFÁTICO": "🩸",
    "PIEL Y ANEXOS CUTÁNEOS": "🧴",
    "ESTUDIOS CITOLÓGICOS ESPECÍFICOS": "🔬",
    "ESTUDIOS ESPECIALES": "❄️",
}

_CATALOG_ROW_RE = re.compile(
    r'^\s*\|\s*[A-Z]{2,4}\d+\s*\|\s*(.+?)\s*\|\s*(\$[\d,.]+)\s*\|'
)
_CATALOG_HEADER_RE = re.compile(r'^## (.+)')


def _format_catalog_as_text(chunks: list[dict]) -> str:
    """Format a complete price list from catalog chunks — no LLM required.

    Parses each chunk's "## SECTION\n| CODE | Description | $PRICE |" structure
    and emits a grouped, Telegram-formatted price list.

    Bypasses the LLM entirely: avoids context-window limits, hallucination, and
    the false "data not loaded" response the LLM produces when given only 1
    item per section.
    """
    section_items: dict[str, list[tuple[str, str]]] = {}
    section_order: list[str] = []

    for chunk in chunks:
        content = chunk["content"]
        header_m = _CATALOG_HEADER_RE.match(content)
        if not header_m:
            continue
        section = header_m.group(1).strip()
        if section not in section_items:
            section_items[section] = []
            section_order.append(section)
        for line in content.split("\n"):
            m = _CATALOG_ROW_RE.match(line)
            if m:
                section_items[section].append((m.group(1).strip(), m.group(2).strip()))

    # Drop sections with no priced procedures (e.g. schedule tables)
    section_order = [s for s in section_order if section_items.get(s)]
    if not section_order:
        return "No encontré información de precios en los documentos."

    lines = [
        "*Lista de Precios — SP Unidad de Diagnóstico Histológico*",
        "_Moneda: USD · Vigente Junio 2026_\n",
    ]
    for section in section_order:
        emoji = _CATALOG_SECTION_EMOJI.get(section.upper(), "🔬")
        lines.append(f"*{section}*")
        for desc, price in section_items[section]:
            lines.append(f"{emoji} {desc} — `{price}`")
        lines.append("")

    return "\n".join(lines).rstrip()


# ─── Generation ──────────────────────────────────────────────────────────────

MIN_SIMILARITY = 0.20  # chunks below this threshold are considered off-topic
LOW_MIN_SIMILARITY = 0.10  # second-pass threshold for approximate matches
VISION_EXTRACT_MAX_TOKENS = 80  # short enough to avoid prose, enough for comma-separated search terms

# ─── Illegible image detection ─────────────────────────────────────────────────

_ILLEGIBLE_PATTERNS = [
    re.compile(r"no\s+puedo\s+(leer|ver|descifrar|descifr|interpretar|distinguir)", re.IGNORECASE),
    re.compile(r"(imagen|foto|imagen)\s+(no\s+)?(ilegible|no\s+es\s+legible|borrosa|oscura|incomprensible)", re.IGNORECASE),
    re.compile(r"(ilegible|incomprensible|indistinguible|indecifrable)", re.IGNORECASE),
    re.compile(r"cannot\s+(read|see|interpret|decipher|make\s+out|determine)", re.IGNORECASE),
    re.compile(r"(image|photo|picture)\s+is\s+(unclear|blurry|dark|illegible|unreadable)", re.IGNORECASE),
    re.compile(r"unable\s+to\s+(read|see|interpret|view|process)\s+(the\s+)?(image|photo|picture)", re.IGNORECASE),
]

# Patterns that indicate PARTIAL legibility — the model could read some text
# but not all. These should NOT trigger the full-illegible fallback; instead
# the LLM is already instructed (via image instruction) to extract what it can.
_PARTIALLY_LEGIBLE_PATTERNS = [
    re.compile(r"(parcialmente|parte|algunas?\s+(partes?|secciones?|palabras?|líneas?|campos?)).*(ilegible|borros[oa]|no\s+puedo|indecifr|dif[ií]cil|indistinguible)", re.IGNORECASE),
    re.compile(r"(ilegible|borros[oa]|indecifrable|no\s+se\s+(lee|ve|puede)).*(parcial|algunas?\s+|parte)", re.IGNORECASE),
    re.compile(r"(puedo\s+)?(leer|ver|distinguir|identificar).*(pero|aunque|sin embargo|no\s+puedo).*(otra|el\s+resto|lo\s+demás|algunas?\s+partes?)", re.IGNORECASE),
    re.compile(r"(some|parts?|sections?).*(illegible|blurry|unreadable|unclear|cannot\s+read)", re.IGNORECASE),
    re.compile(r"(can\s+read|can\s+see|able\s+to\s+read).*(but|however|although).*(some|other|rest|remaining).*(cannot|unreadable|illegible|blurry|unclear)", re.IGNORECASE),
    re.compile(r"(no\s+puedo\s+(leer|descifrar|distinguir)).*(algunas?\s+|ciertos?\s+|parte).*(partes?|secciones?|palabras?|nombres?|montos?)", re.IGNORECASE),
]


def _illegible_fallback_msg(images: list[dict] | None) -> str:
    """Return the appropriate illegible-image message (singular or plural)."""
    if images and len(images) > 1:
        return (
            "No puedo leer las imágenes. La calidad o resolución puede ser insuficiente. "
            "Intentá enviarlas con mejor iluminación o enfoque, o describí tu consulta por texto."
        )
    return (
        "No puedo leer la imagen. La calidad o resolución puede ser insuficiente. "
        "Intentá enviarla con mejor iluminación o enfoque, o describí tu consulta por texto."
    )


def _is_illegible_response(answer: str) -> bool:
    """Check if the vision model's response indicates it couldn't read the image.
    Matches common phrases in both Spanish and English.

    Returns True for fully illegible, False for normal or partially legible.
    Partially legible responses are NOT caught here — the LLM handles them via
    the image instruction (extract what it can, note what's illegible).
    """
    if not answer or len(answer.strip()) < 15:
        return True
    # If partially legible, do NOT treat as fully illegible
    answer_lower = answer.lower()
    if any(p.search(answer_lower) for p in _PARTIALLY_LEGIBLE_PATTERNS):
        return False
    return any(p.search(answer_lower) for p in _ILLEGIBLE_PATTERNS)


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
        reformulated = await call_chat(messages, max_tokens=100, temperature=0.0)
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

_LISTING_QUERY_RE = re.compile(
    r'\b(qu[eé]\s+tipo[s]?|qu[eé]\s+(tienen|tienes)|qu[eé]\s+(estudios|ex[aá]menes|servicios|procedimientos|biopsias|citolog[ií]as|an[aá]lisis)|'
    r'todos\s+los|lista\s+(de|completa)|tienen\s+disponible|qu[eé]\s+ofrecen|cu[aá]les\s+son|'
    r'qu[eé]\s+tipos?\s+de|muestren?\s+(todos|todas)|todo[s]?\s+los\s+(estudios|servicios|ex[aá]menes)|'
    r'de\s+cada\s+uno|de\s+todos\s+los|precio[s]?\s+de\s+(cada|todos))\b',
    re.IGNORECASE,
)

# Detects price intent within a listing query — triggers code-generated full catalog
# (bypasses LLM entirely; always complete, no hallucination, no context limits).
# Bare "precio[s]?" is enough: if user asks for types AND mentions prices, they want prices.
_PRICE_INTENT_RE = re.compile(
    r'\b(precio[s]?|de\s+cada\s+uno|de\s+todos(\s+los)?|todos\s+los\s+precios?|'
    r'lista\s+(completa\s+)?de\s+precios?|precio[s]?\s+de\s+(cada|todos)|'
    r'cu[aá]nto\s+cuesta\s+cada|dame\s+(todos?|todos?\s+los)\s+precios?)\b',
    re.IGNORECASE,
)


async def _hyde_query(question: str, expertise_area: str) -> str:
    """Generate a hypothetical catalog/document answer and return it as the search key.

    Bridges patient-language vs catalog-language vocabulary gap by embedding
    what the answer would look like, not the question itself.
    Returns "" on failure or for broad listing queries — caller falls back to original question.

    Listing queries (e.g. "qué tipos de biopsias tienen") must NOT use HyDE: the prompt
    generates a single procedure name which biases the embedding toward one catalog section,
    suppressing all other categories from the top-k results.
    """
    if _LISTING_QUERY_RE.search(question):
        return ""

    messages = [
        {
            "role": "user",
            "content": (
                f"Sos un especialista en {expertise_area}. Dado el siguiente texto de un cliente, "
                f"escribí el nombre técnico/formal del procedimiento o servicio como aparecería "
                f"en una lista de precios o catálogo de {expertise_area}. "
                f"Usá nomenclatura técnica formal — NO el lenguaje coloquial del cliente. "
                f"NO inventes precios. NO expliques. Respondé SOLO con el nombre técnico "
                f"del procedimiento (1-2 líneas máximo).\n\n"
                f"Texto del cliente: {question}"
            ),
        }
    ]
    try:
        result = await call_chat(messages, max_tokens=60, temperature=0.0)
        result = result.strip()
        if not result or len(result) < 3 or len(result) > 500:
            return ""
        return result
    except Exception as e:
        logger.warning("hyde_query failed: %s — using original query", e)
        return ""


def _chunk_base_term(chunk_content: str) -> str:
    """Extract the base/category term from a table-row chunk's item name.

    Chunk format: "## SECTION\n| CODE | Item Name – Variant | Price |"

    For "Tráquea – Endoscópica" → returns "Tráquea"  (term before " – ")
    For "Apéndice Cecal"        → returns "Apéndice Cecal"  (no separator)

    Using the chunk's own correctly-accented text avoids accent-mismatch
    when the user types "traquea" but the catalog stores "Tráquea".
    """
    lines = chunk_content.split('\n')
    if len(lines) < 2:
        return ""
    cells = [c.strip() for c in lines[1].split('|') if c.strip()]
    if len(cells) < 2:
        return ""
    item_name = cells[1]  # e.g. "Tráquea – Endoscópica"
    for sep in (' – ', ' - ', ' / '):
        if sep in item_name:
            return item_name.split(sep)[0].strip()
    return item_name.strip()


async def _fetch_section_siblings(
    db: AsyncSession,
    namespace: str,
    context_chunks: list[dict],
    question: str,
) -> list[dict]:
    """Fetch sibling table-row chunks from the same catalog section as retrieved chunks.

    HyDE generates a single specific procedure name (e.g. "Tráquea – Endoscópica"),
    biasing retrieval toward one catalog row and potentially missing siblings
    (e.g. "Tráquea – Resección") that share the same base term.

    Search term comes from the RETRIEVED CHUNK's own item name (correctly accented),
    not from the user's question — avoids PostgreSQL ILIKE accent-mismatch where
    "traquea" (user input) does not match "Tráquea" (stored content, á ≠ a).

    Generic: works for any tenant — medical variants, menu sizes, gym plans, etc.
    """
    table_row_chunks = [
        c for c in context_chunks
        if '\n' in c["content"]
        and c["content"].split('\n')[0].startswith('#')
        and c["content"].split('\n')[1].strip().startswith('|')
    ]
    if not table_row_chunks:
        return []

    existing = {c["content"] for c in context_chunks}
    siblings: list[dict] = []

    for chunk in table_row_chunks:
        section_header = chunk["content"].split('\n')[0]
        base_term = _chunk_base_term(chunk["content"])
        if not base_term or len(base_term) < 3:
            continue

        result = await db.execute(
            text("""
                SELECT content, source, page
                FROM document_chunks
                WHERE namespace = :namespace
                  AND embedding IS NOT NULL
                  AND content LIKE :section_prefix
                  AND content ILIKE :kw_pattern
                LIMIT 20
            """),
            {
                "namespace": namespace,
                "section_prefix": section_header + "\n%",
                "kw_pattern": f"%{base_term}%",
            },
        )
        for row in result.fetchall():
            if row.content not in existing:
                siblings.append({
                    "content": row.content,
                    "source": row.source,
                    "page": row.page,
                    "similarity": MIN_SIMILARITY,
                })
                existing.add(row.content)

    return siblings


async def _extract_search_terms_from_images(images: list[dict]) -> str:
    """Use the vision model to extract key search terms from images.

    When a user sends an image with no caption (or a generic default question),
    the vector search has nothing specific to match against. This function
    calls the vision model with a low-token extraction prompt to identify the
    key terms (e.g., study names, medical terms) that can be used as a
    search query for the RAG vector search.

    Returns a search query string, or empty string on failure.
    """
    vision_model = get_setting("llm_vision_model", settings.llm_vision_model)
    if not vision_model:
        return ""

    content: list[dict] = [
        {
            "type": "text",
            "text": (
                "Extraé los términos clave de esta imagen que podrían usarse para "
                "buscar información en una base de datos. Por ejemplo: nombres de "
                "estudios médicos, diagnósticos, o procedimientos. Respondé SOLO con "
                "los términos separados por comas, nada más. Ejemplo: Biopsia de "
                "apéndice cecal, Anexo de apéndice cecal"
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
        result = await call_chat(messages, max_tokens=VISION_EXTRACT_MAX_TOKENS, temperature=0.0, model=vision_model)
        result = result.strip()
        if result:
            logger.info("vision_extracted_terms terms=%r", result[:120])
        return result
    except Exception as e:
        logger.warning("vision_extract_failed: %s", e)
        return ""


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
                                        no_length_limit=no_length_limit)

    if context_chunks:
        context_text = "\n\n---\n\n".join([
            f"[Source: {c['source']}, Page {c['page']}]\n{c['content']}"
            for c in context_chunks
        ])
        confidence_note = "\n\nNOTA: Los siguientes documentos son coincidencias aproximadas (no exactas). Proporcioná la información que encontrés y aclará que puede ser similar pero no idéntico a lo que pregunta el usuario.\n" if low_confidence else ""
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
            "Analizá la(s) imagen(es) que recibiste y respondé basándote en lo que ves. "
            "NUNCA le digas al usuario que envíe una imagen o que contacte por WhatsApp "
            "para enviar una imagen — YA la envió. Si la imagen contiene una orden "
            "médica o documento, extraé la información y respondé con los precios "
            "que encontrés en el contexto. "
            "Si la imagen está PARCIALMENTE legible (podés leer algunas partes pero no todas): "
            "1) Proporcioná la información que SÍ podés leer. "
            "2) Aclará explícitamente qué partes no se pudieron leer (ej: 'No se pudo leer el monto de X'). "
            "3) Sugerí enviar una imagen más clara solo para las partes que no se pudieron leer. "
            "NUNCA descartes toda la imagen si podés leer algo — siempre extraé lo que puedas.]"
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

    vision_model = settings.llm_vision_model or None
    return await call_chat(
        messages,
        max_tokens=max_tokens,
        temperature=0.1,
        channel=channel,
        model=vision_model if images else None,
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

    async with AsyncSessionLocal() as db:
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


async def _tool_search_web(query: str, tenant: "Tenant") -> str:
    """Search the web for a tool dispatch call. Returns formatted string or ''."""
    if not get_setting("web_search_url", settings.web_search_url):
        return ""

    namespace = tenant.slug
    cached = _get_cached(namespace, "search_web", query)
    if cached is not None:
        logger.debug("tool_cache hit: %s:search_web", namespace)
        result, _ = cached
        return result

    web_chunks = await _web_search(query)
    result = "\n\n".join(f"[Web]: {c['content']}" for c in web_chunks) if web_chunks else ""
    _set_cached(namespace, "search_web", query, result, [])
    logger.debug("tool_call: search_web → %d chars", len(result))
    return result


async def _dispatch_tool(tool_call: dict, namespace: str, tenant: "Tenant") -> str:
    """Execute a single tool call and return its string result."""
    name = tool_call["function"]["name"]
    inp = json.loads(tool_call["function"]["arguments"])
    if name == "search_documents":
        result, _ = await _tool_search_documents(inp["query"], namespace, tenant.expertise_area or "")
        return result or "No relevant documents found."
    if name == "search_web":
        return await _tool_search_web(inp["query"], tenant) or "No web results found."
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
        answer = f"¡Hola! ¿En qué puedo ayudarte?{(' ' + expertise_area) if expertise_area else ''}"
        await save_turn(db, user_id, namespace, question, answer, channel=channel, tenant_id=tenant_id)
        return answer, [], "greeting"

    # Pre-RAG: explicit escalation shortcut — skip vector search
    if ESCALATION_PATTERN.search(question):
        area_clause = f" Mi área de expertise: {expertise_area}." if expertise_area else ""
        answer = f"Entiendo que querés hablar con alguien.{area_clause} Contactamos directamente."
        await save_turn(db, user_id, namespace, question, answer, channel=channel)
        await _log_unanswered(db, namespace, question, user_id, "needs_human", tenant_id)
        return answer, [], "needs_human"

    # Vision guard: if user sent an image but no vision model is configured,
    # skip the LLM call entirely — sending image payloads to text-only models
    # produces opaque 404 errors from the provider.
    if images and not get_setting("llm_vision_model", settings.llm_vision_model):
        logger.info("vision_guard ns=%s user=%s — no vision model configured", namespace, user_id)
        answer = "No puedo procesar imágenes en este momento. Por favor, enviá tu consulta por texto."
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
            _system_prompt = build_system_prompt(expertise_area, channel=channel)
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
                    # Collect doc_chunks for source attribution (cache hit — no extra DB call)
                    _doc_chunks: list[dict] = []
                    for tc in tool_calls:
                        if tc["function"]["name"] == "search_documents":
                            q = json.loads(tc["function"]["arguments"])["query"]
                            _, _doc_chunks = await _tool_search_documents(q, namespace, expertise_area or "")
                            break

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
                    return answer, _doc_chunks, "tool_use"

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

    # ── CATALOG PATH ──────────────────────────────────────────────────────────
    # Listing/overview queries need all catalog sections, not just the ones that
    # happen to be most similar to the query terms. Two modes:
    #
    #   • Overview  ("qué tipos tienen")     → LLM + 1 chunk/section (category names)
    #   • Price list ("tipos y precios", "y los precios de cada uno?")
    #       → code-generated from all chunks, no LLM
    #         Bypasses context limits AND the LLM's false "data not loaded" response
    #         that occurs when it receives only 1 item per section.
    #
    # Check both original question AND reformulated search_query because
    # follow-up questions only match after reformulation.
    _is_listing = _LISTING_QUERY_RE.search(question) or _LISTING_QUERY_RE.search(search_query)
    if _is_listing:
        _is_price = _PRICE_INTENT_RE.search(question) or _PRICE_INTENT_RE.search(search_query)
        _use_codegen = _is_price and channel == "telegram"

        if _use_codegen:
            # Code-generated path: fetch all procedures, format in Python, skip LLM.
            full_chunks = await retrieve_full_catalog(db, namespace)
            if full_chunks:
                logger.info(
                    "catalog_codegen ns=%s q=%r items=%d",
                    namespace, question[:60], len(full_chunks),
                )
                answer = _format_catalog_as_text(full_chunks)
                await save_turn(db, user_id, namespace, question, answer,
                                channel=channel, tenant_id=tenant_id)
                return answer, full_chunks, None
        else:
            # LLM overview path: 1 chunk per section → category names (no price intent).
            overview_chunks = await retrieve_catalog_overview(db, namespace)
            if overview_chunks:
                logger.info(
                    "catalog_overview ns=%s q=%r sections=%d",
                    namespace, question[:60], len(overview_chunks),
                )
                answer = await generate_answer(
                    overview_chunks, question, history, expertise_area,
                    channel=channel, images=images,
                )
                answer = validate_output(answer, user_id=user_id)
                if CANARY_TOKEN in answer:
                    answer = answer.replace(CANARY_TOKEN, "[REDACTED]")
                    logger.warning("canary_redacted user_id=%s catalog_overview", user_id)
                await save_turn(db, user_id, namespace, question, answer,
                                channel=channel, tenant_id=tenant_id)
                return answer, overview_chunks, None

        # No section chunks found (unstructured namespace) — fall through to normal path
        logger.info("catalog ns=%s — no section chunks, falling through to similarity", namespace)
    # ── END CATALOG PATH ──────────────────────────────────────────────────────

    # HyDE: embed a hypothetical answer instead of the question to bridge vocabulary gap
    if get_setting("hyde_enabled", "on") == "on":
        hyde_result = await _hyde_query(search_query, expertise_area)
        if hyde_result:
            logger.info("hyde ns=%s q=%r → hypothetical=%r",
                        namespace, search_query[:60], hyde_result[:60])
            search_query = hyde_result

    context = await retrieve_context(db, search_query, namespace)
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

    # Section-sibling completion: when HyDE biases retrieval toward one variant of a
    # catalog item, fetch sibling rows from the same section that match the original
    # question's keywords. Ensures all price variants are visible to the LLM.
    if context:
        siblings = await _fetch_section_siblings(db, namespace, context, question)
        if siblings:
            context = context + siblings
            logger.info(
                "section_siblings ns=%s q=%r added=%d total=%d",
                namespace, question[:60], len(siblings), len(context),
            )

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
            vision_query = await _extract_search_terms_from_images(images)
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

    # If vision-augmented retrieval found context, fall through to the main
    # generate_answer path below. Otherwise, handle the no-context cases.
    if not context:
        # Image-only path (after vision-augmented retrieval also failed to find context):
        # process the images through the vision model without document context.
        if images:
            answer = await generate_answer(
                [], question, history, expertise_area,
                channel=channel, images=images,
            )
            answer = validate_output(answer, user_id=user_id)
            if CANARY_TOKEN in answer:
                answer = answer.replace(CANARY_TOKEN, "[REDACTED]")
                logger.warning("canary_redacted user_id=%s image_only", user_id)
            # Check if vision model couldn't read the image(s)
            if _is_illegible_response(answer):
                answer = _illegible_fallback_msg(images)
                logger.info("vision_illegible ns=%s user=%s images=%d", namespace, user_id, len(images))
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
    )
    answer = validate_output(answer, user_id=user_id)
    # Redact canary token at write time — prevents exfiltration via history
    if CANARY_TOKEN in answer:
        answer = answer.replace(CANARY_TOKEN, "[REDACTED]")
        logger.warning("canary_redacted user_id=%s context_answer", user_id)
    # Check if vision model couldn't read the image(s) (applies when image was sent alongside text context)
    if images and _is_illegible_response(answer):
        answer = _illegible_fallback_msg(images)
        logger.info("vision_illegible ns=%s user=%s images=%d", namespace, user_id, len(images))
    await save_turn(db, user_id, namespace, question, answer, channel=channel)
    return answer, context, None
