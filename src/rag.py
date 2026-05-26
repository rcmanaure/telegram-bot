"""
RAG Pipeline: Embed → Store → Retrieve → Answer

This is the core of the demo. Shows clients:
1. How documents get chunked and embedded
2. How semantic search works (not keyword search)
3. How the LLM answers ONLY from retrieved context (no hallucination)
"""
import logging

import httpx
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession
from db import DocumentChunk, Conversation
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
- Usá listas solo cuando realmente ayude (ej. comparar planes, enumerar horarios).
- Nunca menciones "documentos", "páginas" ni "fuentes" — simplemente sabés la información.
- Podés usar un emoji ocasional, pero menos es más.
- Respondé en el idioma del usuario.
"""

async def _triage_response(question: str, expertise_area: str) -> str:
    """No relevant context found — let the LLM decide how to respond.
    Greetings/chitchat get a warm reply; off-topic questions get a redirect."""
    area = expertise_area or "los temas cubiertos en los documentos"
    messages = [
        {
            "role": "system",
            "content": (
                f"Sos un asistente especializado en: {area}.\n\n"
                "El usuario te envió un mensaje pero no hay información relevante en tu base de conocimiento.\n"
                "Reglas:\n"
                "- Si es un saludo, agradecimiento o mensaje social → respondé de forma amigable "
                "y ofrecé ayuda dentro de tu área de expertise.\n"
                f"- Si es una pregunta fuera de tu área → decile brevemente que tu especialidad es {area} "
                "y pedile que reformule su consulta.\n"
                "- Máximo 2 oraciones. No inventes información técnica ni respondas preguntas fuera de tu área."
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
            json={"model": settings.llm_model, "messages": messages, "max_tokens": 120, "temperature": 0.3},
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except Exception:
        area_clause = f" Mi área de expertise: {expertise_area}." if expertise_area else ""
        return f"Eso está fuera de mi área de expertise.{area_clause} Consultá directamente con nosotros."


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

    if not context_chunks:
        return await _triage_response(question, expertise_area)

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

async def rag_query(
    db: AsyncSession,
    question: str,
    namespace: str,
    user_id: str,
    expertise_area: str = "",
) -> tuple[str, list[dict]]:
    """
    Full RAG pipeline: retrieve context → generate answer → save history.
    Returns (answer, retrieved_chunks) for logging/debugging.
    """
    context = await retrieve_context(db, question, namespace)
    # Drop chunks that are too dissimilar — prevents off-topic questions from
    # reaching the LLM with unrelated context that the model might ignore.
    context = [c for c in context if c["similarity"] >= MIN_SIMILARITY]
    history = await get_history(db, user_id, namespace)
    answer = await generate_answer(context, question, history, expertise_area)
    await save_turn(db, user_id, namespace, question, answer)
    return answer, context
