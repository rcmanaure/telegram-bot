"""
RAG Pipeline: Embed → Store → Retrieve → Answer

This is the core of the demo. Shows clients:
1. How documents get chunked and embedded
2. How semantic search works (not keyword search)
3. How the LLM answers ONLY from retrieved context (no hallucination)
"""
import json
import logging

import httpx
from services.prompts import build_system_prompt, ESCALATION_PATTERN, _GREETING_PATTERN
from services.stt import transcribe_voice
from security import CANARY_TOKEN, scan_chunk_for_injection, sanitize_user_input, validate_output

logger = logging.getLogger(__name__)
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession
from db import DocumentChunk, Conversation, UnansweredQuery, Tenant
from config import settings
from config_overlay import get_setting
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
                "- NEVER say the question is off-topic when it relates to the expertise area. "
                "Questions about the expertise area without document support = 'ambiguous'.\n\n"
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
                "'cuánto cuesta una biopsia?' → ambiguous (relates to expertise area even if no doc found)\n"
                "'como se hace una pizza?' → off_topic\n"
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
        area_clause = f" Nos especializamos en {expertise_area}." if expertise_area else ""
        return "ambiguous", f"Ese tipo de consulta no está dentro de los servicios que ofrecemos.{area_clause} Si necesitás algo relacionado con nuestra área, con gusto te ayudamos."


async def generate_answer(
    context_chunks: list[dict],
    question: str,
    conversation_history: list[dict],
    expertise_area: str = "",
    channel: str = "telegram",
    image_b64: str | None = None,
    image_mime: str = "image/jpeg",
    from_web: bool = False,
) -> str:
    """
    Generate an answer using retrieved context + conversation history.
    When image_b64 is set, sends the image alongside the question (vision models).
    When from_web is True, adds web-source framing to the system prompt.
    """
    if not context_chunks:
        return "No encontré información relevante en los documentos para responder tu pregunta."

    system_prompt = build_system_prompt(expertise_area, channel=channel, from_web=from_web)

    context_text = "\n\n---\n\n".join([
        f"[Source: {c['source']}, Page {c['page']}]\n{c['content']}"
        for c in context_chunks
    ])

    text_content = (
        f"<document_context>\n{context_text}\n</document_context>\n\n"
        f"<user_question>\n{question}\n</user_question>"
    )

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(conversation_history[-6:])

    if image_b64:
        # OpenAI vision format: content is a list with text + image_url parts
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": text_content},
                {"type": "image_url", "image_url": {"url": f"data:{image_mime};base64,{image_b64}"}},
            ],
        })
    else:
        messages.append({"role": "user", "content": text_content})

    vision_model = settings.llm_vision_model or None
    return await call_chat(
        messages,
        max_tokens=800,
        temperature=0.1,
        channel=channel,
        model=vision_model if image_b64 else None,
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
    image_b64: str | None = None,
    image_mime: str = "image/jpeg",
    tenant: "Tenant | None" = None,
) -> tuple[str, list[dict], str | None]:
    """
    Full RAG pipeline: retrieve context → generate answer → save history.
    Returns (answer, retrieved_chunks, intent | None).
    intent is None when answered from docs; otherwise the triage classification.
    When image_b64 is set, the image is passed to generate_answer() for vision models.
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
    if image_b64 and not get_setting("llm_vision_model", settings.llm_vision_model):
        logger.info("vision_guard ns=%s user=%s — no vision model configured", namespace, user_id)
        answer = "No puedo procesar imágenes en este momento. Por favor, enviá tu consulta por texto."
        await save_turn(db, user_id, namespace, question or "📷 [imagen]", answer, channel=channel, tenant_id=tenant_id)
        return answer, [], "no_vision_model"

    # Query reformulation: resolve pronouns/references using conversation history
    # Use reformulated query for vector search, original question for LLM answer generation
    history = await get_history(db, user_id, namespace)
    search_query = await _reformulate_query(question, history)
    if search_query != question:
        logger.info("reformulate ns=%s original=%r → reformulated=%r",
                     namespace, question[:60], search_query[:60])

    context = await retrieve_context(db, search_query, namespace)
    logger.info(
        "retrieve ns=%s q=%r search_q=%r top_scores=%s",
        namespace,
        question[:60],
        search_query[:60] if search_query != question else "(same)",
        [(round(c["similarity"], 3), c["source"], c["content"][:40]) for c in context[:3]],
    )
    context = [c for c in context if c["similarity"] >= MIN_SIMILARITY]

    if not context:
        # Web search fallback: if tenant has web search enabled and URL is configured,
        # try web search before falling back to triage.
        web_search_enabled = tenant.web_search_enabled if tenant else False
        if web_search_enabled and get_setting("web_search_url", settings.web_search_url):
            web_results = await _web_search(question)
            if web_results:
                answer = await generate_answer(
                    web_results, question, history, expertise_area,
                    channel=channel, image_b64=image_b64, image_mime=image_mime,
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
        channel=channel, image_b64=image_b64, image_mime=image_mime,
    )
    answer = validate_output(answer, user_id=user_id)
    # Redact canary token at write time — prevents exfiltration via history
    if CANARY_TOKEN in answer:
        answer = answer.replace(CANARY_TOKEN, "[REDACTED]")
        logger.warning("canary_redacted user_id=%s context_answer", user_id)
    await save_turn(db, user_id, namespace, question, answer, channel=channel)
    return answer, context, None
