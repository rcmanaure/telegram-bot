"""
Tests for web search fallback: _web_search(), rag_query web search path,
describe_image_for_upload(), and prompt web-source framing.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


# ─── _web_search() ───────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_web_search_returns_chunks_on_success():
    """Ollama-style response with 'results' key returns context chunks."""
    from rag import _web_search

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {
        "results": [
            {"body": "Python is a popular programming language used for web development, data science, and automation.", "url": "https://example.com/python"},
            {"body": "Python supports async/await syntax for writing concurrent code using coroutines and event loops.", "url": "https://example.com/async"},
        ]
    }

    def mock_get_setting(key, fallback=""):
        if key == "web_search_url":
            return "https://example.com/api/web_search"
        if key == "llm_api_key":
            return "test-key"
        return fallback

    with patch("rag.http_client") as mock_client, \
         patch("rag.get_setting", side_effect=mock_get_setting):
        mock_client.post = AsyncMock(return_value=mock_response)

        result = await _web_search("what is python")

    assert len(result) == 2
    assert result[0]["source"] == "https://example.com/python"
    assert "Python" in result[0]["content"]
    assert result[0]["similarity"] == 0.4


@pytest.mark.asyncio
async def test_web_search_olls_generic_format():
    """Generic response with 'content' field works as fallback."""
    from rag import _web_search

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {
        "results": [
            {"content": "Rust is a systems programming language focused on safety, speed, and concurrency.", "link": "https://example.com/rust"},
        ]
    }

    def mock_get_setting(key, fallback=""):
        if key == "web_search_url":
            return "https://example.com/api/web_search"
        if key == "llm_api_key":
            return "test-key"
        return fallback

    with patch("rag.http_client") as mock_client, \
         patch("rag.get_setting", side_effect=mock_get_setting):
        mock_client.post = AsyncMock(return_value=mock_response)

        result = await _web_search("what is rust")

    assert len(result) == 1
    assert result[0]["source"] == "https://example.com/rust"


@pytest.mark.asyncio
async def test_web_search_returns_empty_on_timeout():
    """Timeout or network error returns [] — best-effort, never blocks reply."""
    import httpx
    from rag import _web_search

    def mock_get_setting(key, fallback=""):
        if key == "web_search_url":
            return "https://example.com/api/web_search"
        if key == "llm_api_key":
            return "test-key"
        return fallback

    with patch("rag.http_client") as mock_client, \
         patch("rag.get_setting", side_effect=mock_get_setting):
        mock_client.post = AsyncMock(side_effect=httpx.TimeoutException("timeout"))

        result = await _web_search("test query")

    assert result == []


@pytest.mark.asyncio
async def test_web_search_returns_empty_on_no_url():
    """When WEB_SEARCH_URL is empty, returns [] immediately (no HTTP call)."""
    from rag import _web_search

    with patch("rag.get_setting", side_effect=lambda k, f="": ""), \
         patch("rag.settings") as mock_settings:
        mock_settings.web_search_url = ""

        result = await _web_search("test query")

    assert result == []


@pytest.mark.asyncio
async def test_web_search_filters_short_results():
    """Results with content < 50 chars are filtered out."""
    from rag import _web_search

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {
        "results": [
            {"body": "Short.", "url": "https://example.com/short"},  # too short
            {"body": "This is a longer and more detailed result that meets the minimum length threshold for web search results.", "url": "https://example.com/long"},
        ]
    }

    def mock_get_setting(key, fallback=""):
        if key == "web_search_url":
            return "https://example.com/api/web_search"
        if key == "llm_api_key":
            return "test-key"
        return fallback

    with patch("rag.http_client") as mock_client, \
         patch("rag.get_setting", side_effect=mock_get_setting):
        mock_client.post = AsyncMock(return_value=mock_response)

        result = await _web_search("test")

    assert len(result) == 1
    assert "longer" in result[0]["content"]


# ─── rag_query() web search fallback ─────────────────────────────────────────

@pytest.mark.asyncio
async def test_rag_query_web_search_fallback_when_enabled():
    """When no context found and tenant.web_search_enabled=True, falls back to web search."""
    from rag import rag_query

    mock_tenant = MagicMock()
    mock_tenant.web_search_enabled = True
    mock_tenant.id = 1
    mock_tenant.slug = "test-ns"

    web_results = [
        {"content": "Python is a popular programming language used for web development and data science.", "source": "https://example.com", "page": 0, "similarity": 0.5},
    ]

    mock_db = AsyncMock()

    with patch("rag.retrieve_context", AsyncMock(return_value=[])), \
         patch("rag._web_search", AsyncMock(return_value=web_results)) as mock_ws, \
         patch("rag.generate_answer", AsyncMock(return_value="Según información pública: Python is great")), \
         patch("rag.validate_output", side_effect=lambda x, **kw: x), \
         patch("rag.save_turn", AsyncMock()), \
         patch("rag.get_history", AsyncMock(return_value=[])), \
         patch("rag._reformulate_query", AsyncMock(side_effect=lambda q, h: q)), \
         patch("rag.get_setting", side_effect=lambda k, f="": "https://example.com/ws" if k == "web_search_url" else f), \
         patch("rag.settings") as mock_settings, \
         patch("rag._classify_intent", new_callable=AsyncMock, return_value="search_docs"), \
         patch("rag.retrieve_catalog_overview", new_callable=AsyncMock, return_value=[]), \
         patch("rag.is_tool_use_available", return_value=False):
        mock_settings.web_search_url = "https://example.com/ws"

        answer, chunks, intent = await rag_query(
            db=mock_db,
            question="what is python",
            namespace="test-ns",
            user_id="u1",
            tenant=mock_tenant,
        )

    assert intent == "web_search"
    assert len(chunks) == 1
    mock_ws.assert_called_once_with("what is python")


@pytest.mark.asyncio
async def test_rag_query_web_search_disabled_falls_to_triage():
    """When web_search_enabled=False, falls through to _triage_response."""
    from rag import rag_query

    mock_tenant = MagicMock()
    mock_tenant.web_search_enabled = False
    mock_tenant.id = 1

    mock_db = AsyncMock()

    with patch("rag.retrieve_context", AsyncMock(return_value=[])), \
         patch("rag._triage_response", AsyncMock(return_value=("greeting", "¡Hola!"))) as mock_triage, \
         patch("rag.save_turn", AsyncMock()), \
         patch("rag.get_history", AsyncMock(return_value=[])), \
         patch("rag._web_search", AsyncMock()) as mock_ws:

        answer, chunks, intent = await rag_query(
            db=mock_db,
            question="hola",
            namespace="test-ns",
            user_id="u1",
            tenant=mock_tenant,
        )

    assert intent == "greeting"
    mock_ws.assert_not_called()


@pytest.mark.asyncio
async def test_rag_query_web_search_silent_fallback_on_error():
    """When web search fails (timeout), falls through to _triage_response."""
    from rag import rag_query

    mock_tenant = MagicMock()
    mock_tenant.web_search_enabled = True
    mock_tenant.id = 1

    mock_db = AsyncMock()

    with patch("rag.retrieve_context", AsyncMock(return_value=[])), \
         patch("rag._web_search", AsyncMock(return_value=[])), \
         patch("rag._triage_response", AsyncMock(return_value=("off_topic", "Fuera de mi área."))) as mock_triage, \
         patch("rag.save_turn", AsyncMock()), \
         patch("rag._log_unanswered", AsyncMock()), \
         patch("rag.get_history", AsyncMock(return_value=[])), \
         patch("rag._reformulate_query", AsyncMock(side_effect=lambda q, h: q)), \
         patch("rag.validate_output", side_effect=lambda x, **kw: x), \
         patch("rag.get_setting", side_effect=lambda k, f="": "https://example.com/ws" if k == "web_search_url" else f), \
         patch("rag.settings") as mock_settings, \
         patch("rag._classify_intent", new_callable=AsyncMock, return_value="search_docs"), \
         patch("rag.retrieve_catalog_overview", new_callable=AsyncMock, return_value=[]), \
         patch("rag.is_tool_use_available", return_value=False):
        mock_settings.web_search_url = "https://example.com/ws"

        answer, chunks, intent = await rag_query(
            db=mock_db,
            question="random question",
            namespace="test-ns",
            user_id="u1",
            tenant=mock_tenant,
        )

    # Falls through to triage (web search returned empty)
    assert intent == "off_topic"
    assert "Fuera" in answer


@pytest.mark.asyncio
async def test_rag_query_web_search_not_invoked_when_context_found():
    """When context IS found in the KB, web search is never called."""
    from rag import rag_query

    mock_tenant = MagicMock()
    mock_tenant.web_search_enabled = True
    mock_tenant.id = 1

    kb_chunks = [{"content": "info from docs", "source": "doc.pdf", "page": 1, "similarity": 0.85}]

    mock_db = AsyncMock()

    with patch("rag.retrieve_context", AsyncMock(return_value=kb_chunks)), \
         patch("rag._web_search", AsyncMock()) as mock_ws, \
         patch("rag.generate_answer", AsyncMock(return_value="Answer from docs")), \
         patch("rag.validate_output", side_effect=lambda x, **kw: x), \
         patch("rag.save_turn", AsyncMock()), \
         patch("rag.get_history", AsyncMock(return_value=[])), \
         patch("rag._reformulate_query", AsyncMock(side_effect=lambda q, h: q)), \
         patch("rag._classify_intent", new_callable=AsyncMock, return_value="search_docs"), \
         patch("rag.retrieve_catalog_overview", new_callable=AsyncMock, return_value=[]), \
         patch("rag.retrieve_policy_chunks", new_callable=AsyncMock, return_value=[]), \
         patch("rag.retrieve_section_siblings", new_callable=AsyncMock, return_value=[]), \
         patch("rag.is_tool_use_available", return_value=False):

        answer, chunks, intent = await rag_query(
            db=mock_db,
            question="test question",
            namespace="test-ns",
            user_id="u1",
            tenant=mock_tenant,
        )

    assert intent is None  # answered from docs, not triage
    assert chunks[0]["source"] == "doc.pdf"
    mock_ws.assert_not_called()


# ─── build_system_prompt web-source framing ──────────────────────────────────

def test_build_system_prompt_with_web_source():
    """When from_web=True, prompt includes 'Según información pública' framing."""
    from services.prompts import build_system_prompt

    prompt = build_system_prompt("fitness", from_web=True)
    assert "Según información pública" in prompt
    assert "CONTEXTO WEB" in prompt


def test_build_system_prompt_without_web_source():
    """When from_web=False (default), prompt does NOT include web-source framing."""
    from services.prompts import build_system_prompt

    prompt = build_system_prompt("fitness", from_web=False)
    assert "Según información pública" not in prompt
    assert "CONTEXTO WEB" not in prompt


# ─── describe_image_for_upload ──────────────────────────────────────────────

@pytest.mark.asyncio
async def test_describe_image_returns_description():
    """describe_image_for_upload calls vision model and returns description text."""
    from services.upload import describe_image_for_upload

    long_desc = "A photo of a whiteboard with mathematical formulas, equations, and diagrams. The board shows integration by parts and a proof of the fundamental theorem of calculus."

    async def mock_call_chat(messages, **kwargs):
        return long_desc

    with patch("llm.call_chat", side_effect=mock_call_chat), \
         patch("services.upload.settings") as mock_settings:
        mock_settings.llm_vision_model = "llava"

        result = await describe_image_for_upload(b"\xff\xd8\xff\xe0fake_jpg_data", "board.jpg")

    assert "whiteboard" in result
    assert len(result) >= 100  # meets the minimum length threshold


@pytest.mark.asyncio
async def test_describe_image_raises_422_on_empty_response():
    """When vision model returns None, describe_image raises HTTPException 422."""
    from fastapi import HTTPException
    from services.upload import describe_image_for_upload

    with patch("llm.call_chat", AsyncMock(return_value=None)), \
         patch("services.upload.settings") as mock_settings:
        mock_settings.llm_vision_model = "llava"

        with pytest.raises(HTTPException) as exc_info:
            await describe_image_for_upload(b"\xff\xd8\xff\xe0fake_jpg_data", "board.jpg")
        assert exc_info.value.status_code == 422


@pytest.mark.asyncio
async def test_describe_image_rejects_short_description():
    """When vision model returns < 100 chars, describe_image raises HTTPException 422."""
    from fastapi import HTTPException
    from services.upload import describe_image_for_upload

    with patch("llm.call_chat", AsyncMock(return_value="Too short")), \
         patch("services.upload.settings") as mock_settings:
        mock_settings.llm_vision_model = "llava"

        with pytest.raises(HTTPException) as exc_info:
            await describe_image_for_upload(b"\xff\xd8\xff\xe0fake_jpg_data", "board.jpg")
        assert exc_info.value.status_code == 422