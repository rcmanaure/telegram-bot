"""
RAG Pipeline: Embed → Store → Retrieve → Answer

This is the core of the demo. Shows clients:
1. How documents get chunked and embedded
2. How semantic search works (not keyword search)
3. How the LLM answers ONLY from retrieved context (no hallucination)
"""
import json
import logging
import re

import httpx
from security import CANARY_TOKEN, scan_chunk_for_injection, sanitize_user_input, validate_output

logger = logging.getLogger(__name__)
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession
from db import DocumentChunk, Conversation, UnansweredQuery
from config import settings
from llm import call_chat, call_embeddings, extract_json_from_llm_response

http_client = httpx.AsyncClient(timeout=60)


# ─── Chunking ────────────────────────────────────────────────────────────────

def chunk_text(text_content: str, source: str, page: int = 0) -> list[dict]:
    """
    Split text into semantically coherent chunks by splitting on paragraph
    boundaries first, then merging adjacent paragraphs up to chunk_size.
    Falls back to character-based splitting only for oversized paragraphs.
    This keeps section headers with their content so retrieval works correctly.
    """
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

    # Greedily merge adjacent pieces up to chunk_size
    merged: list[str] = []
    current = ''
    for piece in raw_pieces:
        if not current:
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
    """Delete old FAQ chunks and (re)index example_questions for a tenant."""
    await db.execute(
        text("DELETE FROM document_chunks WHERE namespace = :ns AND source = :src"),
        {"ns": tenant_slug, "src": FAQ_SOURCE},
    )
    await db.commit()

    if not example_questions:
        return 0

    chunks = [
        {"content": q.strip(), "source": FAQ_SOURCE, "page": 0}
        for q in example_questions
        if q and q.strip()
    ]
    return await index_chunks(db, chunks, tenant_slug)


# ─── Indexing ────────────────────────────────────────────────────────────────

async def index_chunks(
    db: AsyncSession,
    chunks: list[dict],
    namespace: str,
    auto_commit: bool = True,
) -> int:
    """
    Embed and store chunks in pgvector.
    Returns number of chunks stored.

    When auto_commit=False, the caller is responsible for committing the
    transaction (used for atomic upsert: DELETE old + INSERT new in one commit).
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

    texts = [c["content"] for c in clean_chunks]
    embeddings = await call_embeddings(texts)

    db_chunks = [
        DocumentChunk(
            namespace=namespace,
            source=chunk["source"],
            page=chunk["page"],
            content=chunk["content"],
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

    # pgvector cosine similarity search
    # <=> is cosine distance (lower = more similar)
    result = await db.execute(
        text("""
            SELECT content, source, page,
                   1 - (embedding <=> CAST(:query_vec AS vector)) AS similarity
            FROM document_chunks
            WHERE namespace = :namespace
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


# ─── Generation ──────────────────────────────────────────────────────────────

MIN_SIMILARITY = 0.20  # chunks below this threshold are considered off-topic


def _build_system_prompt(expertise_area: str, channel: str = "telegram") -> str:
    from channels.protocol import CHANNEL_FORMATTING

    area_clause = f" Mi área de expertise: {expertise_area}." if expertise_area else ""
    off_topic_reply = f"Eso está fuera de mi área de expertise.{area_clause} Consultá directamente con nosotros."

    fmt = CHANNEL_FORMATTING.get(channel, CHANNEL_FORMATTING["telegram"])

    return f"""Sos un asistente especializado exclusivamente en la información de los documentos cargados. Tu ÚNICA fuente de conocimiento es el contexto que se te proporciona.

REGLAS INQUEBRANTABLES:
- Si la pregunta no puede responderse con el contexto provisto, respondé exactamente: "{off_topic_reply}"
- NUNCA uses conocimiento general. Matemáticas, programación, cocina, historia, ciencia — todo eso está fuera de tu alcance.
- NUNCA inventes, supongas ni completes información que no esté en el contexto.

Cómo hablar:
- Tono amigable y cercano, sin formalismos corporativos.
- Respondé directo al punto, sin repetir la pregunta.
- Para preguntas simples, una o dos oraciones alcanzan.
- Nunca menciones "documentos", "páginas" ni "fuentes" — simplemente sabés la información.
- Usá emojis temáticos cuando menciones actividades, servicios o conceptos. El emoji va SIEMPRE ANTES del nombre del ítem, elegido por vos según el concepto (ej: 🧪 *Análisis clínicos*, 🔬 *Biopsias*, 🏥 *Consultas*, 💳 *Plan Pro*). Elegí emojis apropiados al contexto del negocio — evitá emojis violentos o clínico-gráficos (como 🔪 para biopsias) y optá por emojis que transmitan cuidado, ciencia y salud (🔬 🧪 🏥 🩺 💊 🧬 📋 ✅ 🩻 🫀). No uses siempre el mismo emoji genérico — elegí el que mejor represente semánticamente cada término.
- NO cierres el mensaje con "¿En qué más puedo ayudarte?" ni "¿Hay algo más en lo que pueda ayudar?" — ya lo dijiste al inicio. Respondé directo y cerrá con la información, sin repetir la oferta de ayuda. Una sola vez al inicio alcanza.
- Respondé en el idioma del usuario.

{fmt.format_instructions}

[CANARY_KEY: {CANARY_TOKEN}]
"""

_ESCALATION_PATTERN = re.compile(
    r'\b(operador|humano|persona real|hablar con alguien|quiero hablar|agente)\b',
    re.IGNORECASE,
)


# ─── LLM calls (delegated to llm.call_chat) ──────────────────────────────────


async def _triage_response(
    question: str,
    expertise_area: str,
    language_code: str | None = None,
) -> tuple[str, str]:
    """Classify intent and generate fallback reply when no context found.
    Returns (intent, reply_text). Intent: greeting | off_topic | needs_human | ambiguous."""
    area = expertise_area or "los temas cubiertos en los documentos"
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
                "- Respond in the same language the user wrote in.\n\n"
                "IMPORTANT: Respond with ONLY a JSON object, no markdown, no preamble:\n"
                '{"intent": "<greeting|off_topic|needs_human|ambiguous>", "reply": "<reply text>"}\n\n'
                "Intent definitions:\n"
                "- greeting: purely social/phatic messages only (hi, hello, thanks, bye, how are you). "
                "NOT questions about capabilities or what you can do.\n"
                "- ambiguous: question that COULD relate to the area but no info was found — "
                "tell the user what topics you can help with and invite them to ask more specifically.\n"
                "- off_topic: question clearly unrelated to your area of expertise.\n"
                "- needs_human: user explicitly wants to speak with a real person.\n\n"
                "Examples:\n"
                "'hi' → greeting\n"
                "'que planes tienes?' → ambiguous (could be about service plans)\n"
                "'que puedes hacer?' → ambiguous (asking about capabilities)\n"
                "'como se hace una pizza?' → off_topic\n"
                "'quiero hablar con un humano' → needs_human"
            ),
        },
        {"role": "user", "content": question},
    ]
    try:
        raw = await call_chat(messages, max_tokens=150, temperature=0.2)
        parsed = extract_json_from_llm_response(raw)
        return parsed["intent"], parsed["reply"]
    except Exception:
        area_clause = f" Mi área de expertise: {expertise_area}." if expertise_area else ""
        return "off_topic", f"Eso está fuera de mi área de expertise.{area_clause} Consultá directamente con nosotros."


async def generate_answer(
    context_chunks: list[dict],
    question: str,
    conversation_history: list[dict],
    expertise_area: str = "",
    channel: str = "telegram",
) -> str:
    """
    Generate an answer using retrieved context + conversation history.
    Uses the configured LLM provider so we can swap models easily.
    """
    if not context_chunks:
        return "No encontré información relevante en los documentos para responder tu pregunta."

    system_prompt = _build_system_prompt(expertise_area, channel=channel)

    # Format context for the prompt
    context_text = "\n\n---\n\n".join([
        f"[Source: {c['source']}, Page {c['page']}]\n{c['content']}"
        for c in context_chunks
    ])

    # system → history → (context + question) LAST
    # LLMs expect the current user turn to be the final message.
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(conversation_history[-6:])
    messages.append({
        "role": "user",
        "content": (
            f"<document_context>\n{context_text}\n</document_context>\n\n"
            f"<user_question>\n{question}\n</user_question>"
        ),
    })

    return await call_chat(messages, max_tokens=800, temperature=0.1, channel=channel)


# ─── Speech-to-Text (Groq Whisper) ───────────────────────────────────────────

async def transcribe_voice(audio_bytes: bytes, filename: str = "voice.ogg") -> str:
    if not settings.groq_api_key:
        raise RuntimeError("GROQ_API_KEY not configured.")
    try:
        response = await http_client.post(
            "https://api.groq.com/openai/v1/audio/transcriptions",
            headers={"Authorization": f"Bearer {settings.groq_api_key}"},
            files={"file": (filename, audio_bytes, "audio/ogg")},
            data={"model": "whisper-large-v3-turbo", "response_format": "text"},
        )
        response.raise_for_status()
        result = response.text.strip()
        logger.debug("transcribe_voice: %d chars for %s", len(result), filename)
        return result
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 429:
            logger.warning("transcribe_voice: Groq 429 rate-limit hit")
            raise RuntimeError("STT service is rate-limited. Try again in a moment.")
        if e.response.status_code == 401:
            logger.warning("transcribe_voice: Groq 401 auth error — check GROQ_API_KEY")
            raise RuntimeError("STT authentication error. Check GROQ_API_KEY.")
        logger.warning("transcribe_voice: Groq %d error body: %s", e.response.status_code, e.response.text)
        raise RuntimeError(f"STT service error ({e.response.status_code}).")
    except httpx.TimeoutException:
        raise RuntimeError("STT service timed out. Please try again.")


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
        elif r.role == "assistant" and CANARY_TOKEN in r.content:
            content = "[message redacted]"
        else:
            content = r.content
        history.append({"role": r.role, "content": content})
    return history


HISTORY_ROW_CAP = 50  # max rows per user per namespace (~25 turns)


async def save_turn(
    db: AsyncSession,
    user_id: str,
    namespace: str,
    user_msg: str,
    assistant_msg: str,
    channel: str = "telegram",
):
    db.add(Conversation(
        user_id=user_id,
        namespace=namespace,
        role="user",
        content=user_msg,
        channel=channel,
    ))
    db.add(Conversation(
        user_id=user_id,
        namespace=namespace,
        role="assistant",
        content=assistant_msg,
        channel=channel,
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


# ─── Full RAG Query (entry point) ────────────────────────────────────────────

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
) -> tuple[str, list[dict], str | None]:
    """
    Full RAG pipeline: retrieve context → generate answer → save history.
    Returns (answer, retrieved_chunks, intent | None).
    intent is None when answered from docs; otherwise the triage classification.
    """
    # Pre-RAG: explicit escalation shortcut — skip vector search
    if _ESCALATION_PATTERN.search(question):
        area_clause = f" Mi área de expertise: {expertise_area}." if expertise_area else ""
        answer = f"Entiendo que querés hablar con alguien.{area_clause} Contactamos directamente."
        await save_turn(db, user_id, namespace, question, answer, channel=channel)
        await _log_unanswered(db, namespace, question, user_id, "needs_human", tenant_id)
        return answer, [], "needs_human"

    context = await retrieve_context(db, question, namespace)
    logger.info(
        "retrieve ns=%s q=%r top_scores=%s",
        namespace,
        question[:60],
        [(round(c["similarity"], 3), c["source"], c["content"][:40]) for c in context[:3]],
    )
    # Drop chunks that are too dissimilar — prevents off-topic questions from
    # reaching the LLM with unrelated context that the model might ignore.
    context = [c for c in context if c["similarity"] >= MIN_SIMILARITY]
    history = await get_history(db, user_id, namespace)

    if not context:
        intent, answer = await _triage_response(question, expertise_area, language_code)
        await save_turn(db, user_id, namespace, question, answer, channel=channel)
        if intent in {"off_topic", "needs_human"}:
            await _log_unanswered(db, namespace, question, user_id, intent, tenant_id)
        return answer, [], intent

    answer = await generate_answer(context, question, history, expertise_area, channel=channel)
    answer = validate_output(answer, user_id=user_id)
    await save_turn(db, user_id, namespace, question, answer, channel=channel)
    return answer, context, None
