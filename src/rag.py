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
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession
from db import DocumentChunk, Conversation, UnansweredQuery
from config import settings

openai_client = AsyncOpenAI(
    api_key=settings.openrouter_api_key,
    base_url="https://openrouter.ai/api/v1",
    max_retries=2,
    timeout=30.0,
)

http_client = httpx.AsyncClient(timeout=60)


# ─── Chunking ────────────────────────────────────────────────────────────────

def chunk_text(text_content: str, source: str, page: int = 0) -> list[dict]:
    """
    Split text into overlapping chunks for embedding.
    Overlap ensures context isn't lost at chunk boundaries.
    """
    chunks = []
    size = settings.chunk_size
    overlap = settings.chunk_overlap
    start = 0

    while start < len(text_content):
        end = start + size
        chunk = text_content[start:end].strip()

        if len(chunk) > 50:  # skip tiny chunks
            chunks.append({
                "content": chunk,
                "source": source,
                "page": page,
            })

        start = end - overlap  # overlap with previous chunk

    return chunks


# ─── Embeddings ──────────────────────────────────────────────────────────────

async def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Embed a list of texts via OpenRouter.
    Returns list of 1536-dim vectors. Batched max 100 per call.
    """
    from openai import APIError, APITimeoutError, RateLimitError
    all_embeddings = []
    batch_size = 100

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        try:
            response = await openai_client.embeddings.create(
                model=settings.embedding_model,
                input=batch,
            )
            all_embeddings.extend([item.embedding for item in response.data])
        except RateLimitError:
            raise RuntimeError("Embedding service is rate-limited. Try again in a moment.")
        except APITimeoutError:
            raise RuntimeError("Embedding service timed out. Check your OpenRouter API key and quota.")
        except APIError as e:
            raise RuntimeError(f"Embedding failed: {e.message}")

    return all_embeddings


# ─── Indexing ────────────────────────────────────────────────────────────────

async def index_chunks(
    db: AsyncSession,
    chunks: list[dict],
    namespace: str,
) -> int:
    """
    Embed and store chunks in pgvector.
    Returns number of chunks stored.
    """
    if not chunks:
        return 0

    texts = [c["content"] for c in chunks]
    embeddings = await embed_texts(texts)

    db_chunks = [
        DocumentChunk(
            namespace=namespace,
            source=chunk["source"],
            page=chunk["page"],
            content=chunk["content"],
            embedding=embedding,
        )
        for chunk, embedding in zip(chunks, embeddings)
    ]

    db.add_all(db_chunks)
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
    query_embedding = (await embed_texts([query]))[0]

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

MIN_SIMILARITY = 0.30  # chunks below this threshold are considered off-topic


def _build_system_prompt(expertise_area: str) -> str:
    area_clause = f" Mi área de expertise: {expertise_area}." if expertise_area else ""
    off_topic_reply = f"Eso está fuera de mi área de expertise.{area_clause} Consultá directamente con nosotros."
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
- Usá emojis temáticos cuando menciones actividades, servicios o conceptos. El emoji va SIEMPRE ANTES del nombre del ítem, elegido por vos según el concepto (ej: 🧘‍♀️ *Yoga*, 🚴 *Cycling*, 💪 *Entrenamiento*, 💳 *Plan Pro*). No uses siempre el mismo emoji genérico — elegí el que mejor represente semánticamente cada término. Telegram tiene una gama enorme; aprovechala para hacer el mensaje más visual y fácil de escanear.
- Respondé en el idioma del usuario.

Formato para Telegram (OBLIGATORIO):
- NUNCA uses tablas Markdown (| col | col |) — Telegram no las renderiza, se ven como texto crudo.
- Para comparar opciones o listar ítems con atributos usá listas con viñetas y negrita:
  *Plan Basic* — $29/mes · acceso 6am–10pm · 2 clases/mes
  *Plan Pro* — $59/mes · acceso 24/7 · clases ilimitadas
- Para listas simples usá guiones o números.
- Para horarios o datos tabulares usá formato vertical:
  📅 Yoga: Lun/Mié/Vie — 7am y 6pm
  📅 HIIT: Mar/Jue — 6am y 7pm
- Negrita con *asteriscos* para títulos o datos clave.
- Código con `backticks` solo para datos técnicos exactos (precios, horarios puntuales).
"""

_ESCALATION_PATTERN = re.compile(
    r'\b(operador|humano|persona real|hablar con alguien|quiero hablar|agente)\b',
    re.IGNORECASE,
)


async def _triage_response(
    question: str,
    expertise_area: str,
    language_code: str | None = None,
) -> tuple[str, str]:
    """Classify intent and generate fallback reply when no context found.
    Returns (intent, reply_text). Intent: greeting | off_topic | needs_human | ambiguous."""
    area = expertise_area or "los temas cubiertos en los documentos"
    lang_hint = ""
    if language_code and not language_code.startswith("es"):
        lang_hint = f"\nRespond in the user's language (language_code={language_code})."
    messages = [
        {
            "role": "system",
            "content": (
                f"You are an assistant specialized in: {area}.\n"
                "The user sent a message but there is no relevant information in your knowledge base.\n"
                "Classify the intent and reply briefly (max 2 sentences).\n"
                f"{lang_hint}\n"
                "IMPORTANT: Respond with ONLY a JSON object, no markdown, no preamble:\n"
                '{"intent": "<greeting|off_topic|needs_human|ambiguous>", "reply": "<your reply text>"}\n\n'
                "Intent definitions:\n"
                "- greeting: social message (hi, thanks, bye)\n"
                "- off_topic: question outside your area of expertise\n"
                "- needs_human: user explicitly wants to speak with a person\n"
                "- ambiguous: unclear if related to your area"
            ),
        },
        {"role": "user", "content": question},
    ]
    try:
        response = await http_client.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {settings.openrouter_api_key}",
                "HTTP-Referer": "https://github.com/ruben-portfolio",
            },
            json={"model": settings.llm_model, "messages": messages, "max_tokens": 150, "temperature": 0.2},
        )
        response.raise_for_status()
        raw = response.json()["choices"][0]["message"]["content"]
        parsed = json.loads(raw)
        return parsed["intent"], parsed["reply"]
    except Exception:
        area_clause = f" Mi área de expertise: {expertise_area}." if expertise_area else ""
        return "off_topic", f"Eso está fuera de mi área de expertise.{area_clause} Consultá directamente con nosotros."


async def generate_answer(
    context_chunks: list[dict],
    question: str,
    conversation_history: list[dict],
    expertise_area: str = "",
) -> str:
    """
    Generate an answer using retrieved context + conversation history.
    Uses OpenRouter so we can swap models easily.
    """
    system_prompt = _build_system_prompt(expertise_area)

    # Format context for the prompt
    context_text = "\n\n---\n\n".join([
        f"[Source: {c['source']}, Page {c['page']}]\n{c['content']}"
        for c in context_chunks
    ])

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": f"DOCUMENT CONTEXT:\n{context_text}"
        },
        {
            "role": "assistant",
            "content": "I have read the context. I will only answer based on this information."
        },
    ]

    # Add last 6 turns of conversation history for context
    messages.extend(conversation_history[-6:])

    # Add current question
    messages.append({"role": "user", "content": question})

    try:
        response = await http_client.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {settings.openrouter_api_key}",
                "HTTP-Referer": "https://github.com/ruben-portfolio",
            },
            json={
                "model": settings.llm_model,
                "messages": messages,
                "max_tokens": 800,
                "temperature": 0.1,
            }
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 429:
            raise RuntimeError("LLM service is rate-limited. Please try again in a moment.")
        body = e.response.json() if e.response.headers.get("content-type", "").startswith("application/json") else e.response.text
        msg = body.get("error", {}).get("message", str(body)) if isinstance(body, dict) else body
        raise RuntimeError(f"LLM service error ({e.response.status_code}): {msg}")
    except httpx.TimeoutException:
        raise RuntimeError("LLM service timed out. Please try again.")
    except (KeyError, IndexError):
        raise RuntimeError("Unexpected response from LLM service.")


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
    return [{"role": r.role, "content": r.content} for r in rows]


HISTORY_ROW_CAP = 50  # max rows per user per namespace (~25 turns)


async def save_turn(
    db: AsyncSession,
    user_id: str,
    namespace: str,
    user_msg: str,
    assistant_msg: str,
):
    db.add(Conversation(
        user_id=user_id,
        namespace=namespace,
        role="user",
        content=user_msg,
    ))
    db.add(Conversation(
        user_id=user_id,
        namespace=namespace,
        role="assistant",
        content=assistant_msg,
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
) -> tuple[str, list[dict], str | None]:
    """
    Full RAG pipeline: retrieve context → generate answer → save history.
    Returns (answer, retrieved_chunks, intent | None).
    intent is None when answered from docs; otherwise the triage classification.
    """
    # Pre-RAG: explicit escalation shortcut — skip vector search
    if _ESCALATION_PATTERN.search(question):
        area_clause = f" Mi área de expertise: {expertise_area}." if expertise_area else ""
        answer = f"Entiendo que querés hablar con alguien.{area_clause} Contactanos directamente."
        await save_turn(db, user_id, namespace, question, answer)
        await _log_unanswered(db, namespace, question, user_id, "needs_human", tenant_id)
        return answer, [], "needs_human"

    context = await retrieve_context(db, question, namespace)
    # Drop chunks that are too dissimilar — prevents off-topic questions from
    # reaching the LLM with unrelated context that the model might ignore.
    context = [c for c in context if c["similarity"] >= MIN_SIMILARITY]
    history = await get_history(db, user_id, namespace)

    if not context:
        intent, answer = await _triage_response(question, expertise_area, language_code)
        await save_turn(db, user_id, namespace, question, answer)
        if intent in {"off_topic", "needs_human"}:
            await _log_unanswered(db, namespace, question, user_id, intent, tenant_id)
        return answer, [], intent

    answer = await generate_answer(context, question, history, expertise_area)
    await save_turn(db, user_id, namespace, question, answer)
    return answer, context, None
