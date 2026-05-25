"""
RAG Pipeline: Embed → Store → Retrieve → Answer

This is the core of the demo. Shows clients:
1. How documents get chunked and embedded
2. How semantic search works (not keyword search)
3. How the LLM answers ONLY from retrieved context (no hallucination)
"""
import httpx
from openai import AsyncOpenAI
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

SYSTEM_PROMPT = """Eres un asistente útil que responde preguntas estrictamente basándose en el contexto de documentos proporcionado.

Reglas:
- USA ÚNICAMENTE la información del contexto provisto. Nunca inventes información.
- Si el contexto no contiene suficiente información para responder, indícalo claramente.
- Cita el documento o sección que estás referenciando (ej. "Según [nombre de archivo], página X...")
- Sé conciso y directo. Usa viñetas al listar varios elementos.
- Responde en español por defecto. Si el usuario escribe en otro idioma, responde en ese idioma.
"""

async def generate_answer(
    context_chunks: list[dict],
    question: str,
    conversation_history: list[dict],
) -> str:
    """
    Generate an answer using retrieved context + conversation history.
    Uses OpenRouter so we can swap models easily.
    """
    if not context_chunks:
        return (
            "No encontré información relevante en los documentos para responder tu pregunta. "
            "Intentá reformularla o preguntá sobre otro tema cubierto en los documentos cargados."
        )

    # Format context for the prompt
    context_text = "\n\n---\n\n".join([
        f"[Source: {c['source']}, Page {c['page']}]\n{c['content']}"
        for c in context_chunks
    ])

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
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
    telegram_user_id: str,
    namespace: str,
    limit: int = 10,
) -> list[dict]:
    result = await db.execute(
        select(Conversation)
        .where(
            Conversation.telegram_user_id == telegram_user_id,
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
    telegram_user_id: str,
    namespace: str,
    user_msg: str,
    assistant_msg: str,
):
    db.add(Conversation(
        telegram_user_id=telegram_user_id,
        namespace=namespace,
        role="user",
        content=user_msg,
    ))
    db.add(Conversation(
        telegram_user_id=telegram_user_id,
        namespace=namespace,
        role="assistant",
        content=assistant_msg,
    ))
    await db.commit()

    # Trim old rows, keeping only the most recent HISTORY_ROW_CAP entries
    await db.execute(
        text("""
            DELETE FROM conversations
            WHERE telegram_user_id = :uid AND namespace = :ns
            AND id NOT IN (
                SELECT id FROM conversations
                WHERE telegram_user_id = :uid AND namespace = :ns
                ORDER BY created_at DESC
                LIMIT :cap
            )
        """),
        {"uid": telegram_user_id, "ns": namespace, "cap": HISTORY_ROW_CAP},
    )
    await db.commit()


# ─── Full RAG Query (entry point) ────────────────────────────────────────────

async def rag_query(
    db: AsyncSession,
    question: str,
    namespace: str,
    telegram_user_id: str,
) -> tuple[str, list[dict]]:
    """
    Full RAG pipeline: retrieve context → generate answer → save history.
    Returns (answer, retrieved_chunks) for logging/debugging.
    """
    # 1. Retrieve relevant context
    context = await retrieve_context(db, question, namespace)

    # 2. Load conversation history
    history = await get_history(db, telegram_user_id, namespace)

    # 3. Generate answer
    answer = await generate_answer(context, question, history)

    # 4. Save to history
    await save_turn(db, telegram_user_id, namespace, question, answer)

    return answer, context
