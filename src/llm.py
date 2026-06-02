"""
Provider-agnostic LLM layer.

Supports any OpenAI-compatible chat/completion and embedding API:
- Ollama Cloud (https://ollama.com/v1)
- OpenRouter (https://openrouter.ai/api/v1)
- Local Ollama (http://localhost:11434/v1)
- Any other OpenAI-compatible endpoint

Chat and embedding providers are configured independently via env vars:
  LLM_BASE_URL, LLM_API_KEY, LLM_MODEL, LLM_FALLBACK_MODEL
  EMBEDDING_BASE_URL, EMBEDDING_API_KEY, EMBEDDING_MODEL, EMBEDDING_DIM
"""

import json
import logging
import re
import time

import httpx
from openai import AsyncOpenAI, APIError, APITimeoutError, RateLimitError

from config import settings
from config_overlay import get_setting, get_setting_int

logger = logging.getLogger(__name__)

# ─── Module-level clients ────────────────────────────────────────────────────

_chat_client = httpx.AsyncClient(
    timeout=60.0,
    limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
    transport=httpx.AsyncHTTPTransport(retries=2),
)

# Embedding client is lazy-initialized to avoid crashing at import time
# when EMBEDDING_API_KEY is not set (e.g., during tests without .env).
_embedding_client: AsyncOpenAI | None = None


def _get_embedding_client() -> AsyncOpenAI:
    """Get or create the embedding OpenAI client."""
    global _embedding_client
    if _embedding_client is None:
        _embedding_client = AsyncOpenAI(
            api_key=get_setting("embedding_api_key", settings.effective_embedding_api_key),
            base_url=get_setting("embedding_base_url", settings.embedding_base_url),
            max_retries=2,
            timeout=30.0,
        )
    return _embedding_client


def reset_embedding_client() -> None:
    """Force recreation of embedding client on next call (after config overlay change)."""
    global _embedding_client
    _embedding_client = None


# ─── Error messages ──────────────────────────────────────────────────────────

def _error_message(exc: Exception) -> str:
    """Convert an LLM/embedding exception into a user-facing RuntimeError message."""
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        if status == 402:
            return "LLM service payment required. Check your provider plan and quota."
        if status == 429:
            return "LLM service is rate-limited. Please try again in a moment."
        body = exc.response.json() if exc.response.headers.get("content-type", "").startswith("application/json") else exc.response.text
        msg = body.get("error", {}).get("message", str(body)) if isinstance(body, dict) else body
        return f"LLM service error ({status}): {msg}"
    if isinstance(exc, httpx.TimeoutException):
        return "LLM service timed out. Please try again."
    return "Unexpected response from LLM service."


# ─── Chat completions ─────────────────────────────────────────────────────────

async def call_chat(
    messages: list[dict],
    max_tokens: int = 800,
    temperature: float = 0.1,
    channel: str = "telegram",
    model: str | None = None,
) -> str:
    """
    Call chat/completions with primary model, then fallback on failure.
    Returns the assistant message content string.
    Raises RuntimeError on all-model failure.

    When model= is set (e.g. for vision), only that model is tried — no fallback.
    Text-only fallback models must not receive image payloads.
    """
    if model:
        models = [model]
    else:
        models = [get_setting("llm_model", settings.llm_model)]
        fallback = get_setting("llm_fallback_model", settings.llm_fallback_model)
        if fallback:
            models.append(fallback)

    # Conditional HTTP-Referer header: only sent when using OpenRouter
    # (E2: other providers may reject or ignore it)
    base_url = get_setting("llm_base_url", settings.llm_base_url)
    api_key = get_setting("llm_api_key", settings.effective_llm_api_key)
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if "openrouter.ai" in base_url:
        headers["HTTP-Referer"] = "https://github.com/ruben-portfolio"

    last_error: Exception | None = None
    for model in models:
        t0 = time.monotonic()
        try:
            response = await _chat_client.post(
                f"{base_url}/chat/completions",
                headers=headers,
                json={
                    "model": model,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                },
            )
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
            elapsed_ms = (time.monotonic() - t0) * 1000
            logger.info(
                "llm_call provider=%s model=%s latency_ms=%.0f ok=true",
                base_url, model, elapsed_ms,
            )
            primary_model = get_setting("llm_model", settings.llm_model)
            if model != primary_model:
                logger.warning(
                    "llm_failover primary=%s fallback=%s error_type=%s latency_ms=%.0f",
                    primary_model, model, type(last_error).__name__ if last_error else "unknown", elapsed_ms,
                )
            return content
        except (httpx.HTTPError, KeyError, IndexError, json.JSONDecodeError) as e:
            elapsed_ms = (time.monotonic() - t0) * 1000
            logger.warning(
                "llm_call_failed provider=%s model=%s error=%s latency_ms=%.0f",
                settings.llm_base_url, model, type(e).__name__, elapsed_ms,
            )
            last_error = e
            continue

    # All models failed
    logger.error(
        "llm_all_failed provider=%s models=%s",
        base_url, models,
    )
    raise RuntimeError(_error_message(last_error))


# ─── Embeddings ───────────────────────────────────────────────────────────────

async def call_embeddings(texts: list[str]) -> list[list[float]]:
    """
    Embed a list of texts via the configured embedding provider.
    Returns list of vectors with dimension matching EMBEDDING_DIM.
    Batches in groups of 100 to avoid API limits.
    """
    if not texts:
        return []

    all_embeddings: list[list[float]] = []
    batch_size = 100

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        try:
            response = await _get_embedding_client().embeddings.create(
                model=get_setting("embedding_model", settings.embedding_model),
                input=batch,
            )
            all_embeddings.extend([item.embedding for item in response.data])
        except RateLimitError:
            raise RuntimeError("Embedding service is rate-limited. Try again in a moment.")
        except APITimeoutError:
            raise RuntimeError("Embedding service timed out. Check your embedding provider API key and quota.")
        except APIError as e:
            raise RuntimeError(f"Embedding failed: {e.message}")

    return all_embeddings


# ─── Connection test ──────────────────────────────────────────────────────────

async def test_connection(
    base_url: str,
    api_key: str,
    model: str,
) -> dict:
    """
    Send a tiny chat request to verify provider config.
    Returns {"ok": True, "model": ..., "latency_ms": ...} on success.
    Returns {"ok": False, "error": ...} on failure.
    """
    try:
        t0 = time.monotonic()
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": "ping"}],
                    "max_tokens": 1,
                },
            )
            response.raise_for_status()
            elapsed_ms = (time.monotonic() - t0) * 1000
            return {"ok": True, "model": model, "latency_ms": round(elapsed_ms)}
    except httpx.HTTPStatusError as e:
        return {"ok": False, "error": f"HTTP {e.response.status_code}: {e.response.text[:200]}"}
    except httpx.ConnectError:
        return {"ok": False, "error": f"Cannot connect to {base_url}. Check the URL and network."}
    except httpx.TimeoutException:
        return {"ok": False, "error": f"Connection timed out to {base_url}."}
    except Exception as e:
        return {"ok": False, "error": str(e)[:200]}


# ─── JSON extraction ─────────────────────────────────────────────────────────

def extract_json_from_llm_response(text: str) -> dict:
    """
    Extract a JSON object from an LLM response.
    Handles:
    - Bare JSON: {"intent": "off_topic", "reply": "..."}
    - Markdown-wrapped: ```json\n{...}\n```
    - Extra text around the JSON

    Returns the parsed dict.
    Raises ValueError if no JSON object found.
    """
    # Try stripping markdown code fences first
    # Match ```json ... ``` or ``` ... ``` blocks
    fence_pattern = re.compile(r"```(?:json)?\s*\n?(.*?)\n?\s*```", re.DOTALL)
    fence_match = fence_pattern.search(text)
    if fence_match:
        candidate = fence_match.group(1).strip()
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass  # Fall through to brute-force search

    # Try the whole text as JSON
    text_stripped = text.strip()
    try:
        return json.loads(text_stripped)
    except json.JSONDecodeError:
        pass

    # Brute-force: find the first { ... } in the text
    brace_pattern = re.compile(r"\{[^{}]*\}", re.DOTALL)
    brace_match = brace_pattern.search(text)
    if brace_match:
        try:
            return json.loads(brace_match.group(0))
        except json.JSONDecodeError:
            pass

    raise ValueError(f"No JSON object found in LLM response: {text[:100]}...")


# ─── Startup validation ──────────────────────────────────────────────────────

async def validate_config() -> None:
    """
    Validate LLM and embedding configuration at startup.
    - Check that embedding dimensions match EMBEDDING_DIM.
    - Warn if EMBEDDING_BASE_URL points to Ollama Cloud (no embeddings).
    - Log config summary.
    """
    llm_url = get_setting("llm_base_url", settings.llm_base_url)
    llm_m = get_setting("llm_model", settings.llm_model)
    llm_fb = get_setting("llm_fallback_model", settings.llm_fallback_model)
    emb_url = get_setting("embedding_base_url", settings.embedding_base_url)
    emb_m = get_setting("embedding_model", settings.embedding_model)
    emb_dim = get_setting_int("embedding_dim", settings.embedding_dim)

    logger.info(
        "LLM config: provider=%s model=%s fallback=%s",
        llm_url, llm_m, llm_fb or "(disabled)",
    )
    logger.info(
        "Embedding config: provider=%s model=%s dim=%d",
        emb_url, emb_m, emb_dim,
    )

    # Warn if EMBEDDING_BASE_URL points to Ollama Cloud (no embedding support)
    if "ollama.com" in emb_url:
        logger.warning(
            "EMBEDDING_BASE_URL (%s) points to Ollama Cloud, which does not support embeddings. "
            "Set EMBEDDING_BASE_URL to an OpenAI-compatible embedding endpoint.",
            emb_url,
        )

    # Check embedding dimension
    emb_api_key = get_setting("embedding_api_key", settings.effective_embedding_api_key)
    if not emb_api_key:
        logger.warning(
            "EMBEDDING_API_KEY not set — skipping dimension check. "
            "Set EMBEDDING_API_KEY or OPENROUTER_API_KEY to enable embeddings."
        )
        return

    try:
        result = await call_embeddings(["dimension check"])
        actual_dim = len(result[0])
        if actual_dim != emb_dim:
            logger.error(
                "Embedding dimension mismatch: model produces %d-dim vectors, "
                "but EMBEDDING_DIM=%d. Check EMBEDDING_MODEL and EMBEDDING_DIM.",
                actual_dim, emb_dim,
            )
        else:
            logger.info("Embedding dimension check passed: %d-dim vectors match EMBEDDING_DIM=%d", actual_dim, emb_dim)
    except Exception as e:
        logger.error("Embedding dimension check failed: %s", e)


# ─── Shutdown hook ────────────────────────────────────────────────────────────

async def close_llm_clients() -> None:
    """Gracefully close HTTP clients. Call during app shutdown."""
    await _chat_client.aclose()
    if _embedding_client is not None:
        await _embedding_client.close()