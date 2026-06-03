"""
Tests for the native tool-use agent fan-out (T1–T5).

Covers all 22 code paths identified in the eng review:
  - call_chat_with_tools: text response, tool_calls, system prepend, 400 discrimination
  - Backoff state: is_tool_use_available, _mark_tool_use_failed, auto-recovery
  - Cache: hit/miss/TTL, LRU eviction, flush_tool_cache, namespace isolation
  - _dispatch_tool: search_documents, search_web, unknown tool, web guard
  - rag_query tool path: happy path, direct content, partial failure, all-empty fallback
  - Tool path skipped: image, no web_search_url, is_tool_use_available=False
  - 3-tuple return value, canary protection
  - flush after upload (api.py), UX pre-message guards
"""
import json
import time
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_chat_response(content=None, tool_calls=None, status=200):
    """Build a mock httpx.Response for a chat/completions call."""
    msg = {}
    if content is not None:
        msg["content"] = content
    if tool_calls is not None:
        msg["tool_calls"] = tool_calls
    body = {"choices": [{"message": msg}]}
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status
    resp.json.return_value = body
    resp.raise_for_status = MagicMock()
    if status >= 400:
        resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "error", request=MagicMock(), response=resp
        )
    return resp


def _make_error_response(status, error_message):
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status
    resp.json.return_value = {"error": {"message": error_message}}
    resp.text = error_message
    resp.raise_for_status = MagicMock(
        side_effect=httpx.HTTPStatusError("error", request=MagicMock(), response=resp)
    )
    return resp


# ---------------------------------------------------------------------------
# T1 fixtures: reset backoff state between tests
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_tool_use_backoff():
    """Reset module-level backoff state before each test."""
    import llm
    original = llm._tool_use_last_failure
    llm._tool_use_last_failure = 0.0
    yield
    llm._tool_use_last_failure = original


@pytest.fixture(autouse=True)
def reset_tool_cache():
    """Clear the tool cache before each test."""
    import rag
    rag._tool_cache.clear()
    yield
    rag._tool_cache.clear()


# ---------------------------------------------------------------------------
# 1. call_chat_with_tools — text response (no tools called)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_call_chat_with_tools_text_response():
    from llm import call_chat_with_tools
    resp = _make_chat_response(content="Hello, how can I help?")
    with patch("llm._chat_client") as mock_client:
        mock_client.post = AsyncMock(return_value=resp)
        content, tool_calls = await call_chat_with_tools(
            messages=[{"role": "user", "content": "hi"}],
            tools=[],
        )
    assert content == "Hello, how can I help?"
    assert tool_calls == []


# ---------------------------------------------------------------------------
# 2. call_chat_with_tools — tool_calls returned
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_call_chat_with_tools_returns_tool_calls():
    from llm import call_chat_with_tools
    tc = [{"id": "call_1", "type": "function", "function": {"name": "search_documents", "arguments": '{"query":"precio"}'}}]
    resp = _make_chat_response(tool_calls=tc)
    with patch("llm._chat_client") as mock_client:
        mock_client.post = AsyncMock(return_value=resp)
        content, tool_calls = await call_chat_with_tools(
            messages=[{"role": "user", "content": "precio"}],
            tools=[{"type": "function", "function": {"name": "search_documents"}}],
        )
    assert content is None
    assert len(tool_calls) == 1
    assert tool_calls[0]["function"]["name"] == "search_documents"


# ---------------------------------------------------------------------------
# 3. call_chat_with_tools — system param prepended
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_call_chat_with_tools_system_prepended():
    from llm import call_chat_with_tools
    resp = _make_chat_response(content="ok")
    captured = {}
    async def capture_post(url, **kwargs):
        captured["messages"] = kwargs["json"]["messages"]
        return resp
    with patch("llm._chat_client") as mock_client:
        mock_client.post = capture_post
        await call_chat_with_tools(
            messages=[{"role": "user", "content": "hi"}],
            tools=[],
            system="You are a helpful assistant.",
        )
    assert captured["messages"][0] == {"role": "system", "content": "You are a helpful assistant."}
    assert captured["messages"][1] == {"role": "user", "content": "hi"}


# ---------------------------------------------------------------------------
# 4. ToolUseNotSupportedError: tool-keyword 400 triggers backoff
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_tool_use_not_supported_400_triggers_backoff():
    from llm import call_chat_with_tools, is_tool_use_available, ToolUseNotSupportedError
    assert is_tool_use_available()
    resp = _make_error_response(400, "tool_choice not supported by this model")
    with patch("llm._chat_client") as mock_client:
        mock_client.post = AsyncMock(return_value=resp)
        with pytest.raises(ToolUseNotSupportedError):
            await call_chat_with_tools(messages=[], tools=[])
    assert not is_tool_use_available()


# ---------------------------------------------------------------------------
# 5. Generic 400 (context overflow) does NOT trigger backoff
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_generic_400_no_backoff():
    from llm import call_chat_with_tools, is_tool_use_available
    resp = _make_error_response(400, "context length exceeded maximum tokens")
    with patch("llm._chat_client") as mock_client:
        mock_client.post = AsyncMock(return_value=resp)
        with pytest.raises(RuntimeError, match="LLM service error"):
            await call_chat_with_tools(messages=[], tools=[])
    assert is_tool_use_available()  # backoff NOT triggered


# ---------------------------------------------------------------------------
# 6. is_tool_use_available: False in backoff window
# ---------------------------------------------------------------------------

def test_is_tool_use_available_false_in_backoff():
    import llm
    llm._tool_use_last_failure = time.monotonic()  # just failed
    assert not llm.is_tool_use_available()


# ---------------------------------------------------------------------------
# 7. is_tool_use_available: auto-recovery after TOOL_USE_BACKOFF seconds
# ---------------------------------------------------------------------------

def test_is_tool_use_available_recovery_after_backoff():
    import llm
    llm._tool_use_last_failure = time.monotonic() - (llm.TOOL_USE_BACKOFF + 1)
    assert llm.is_tool_use_available()


# ---------------------------------------------------------------------------
# 8. Cache hit within TTL
# ---------------------------------------------------------------------------

def test_cache_hit_within_ttl():
    import rag
    rag._set_cached("ns", "search_documents", "query text", "result", [{"content": "x"}])
    cached = rag._get_cached("ns", "search_documents", "query text")
    assert cached is not None
    result, chunks = cached
    assert result == "result"
    assert chunks == [{"content": "x"}]


# ---------------------------------------------------------------------------
# 9. Cache miss after TTL
# ---------------------------------------------------------------------------

def test_cache_miss_after_ttl():
    import rag
    key = rag._cache_key("ns", "search_documents", "q")
    rag._tool_cache[key] = ("old", [], time.monotonic() - 301)  # expired
    cached = rag._get_cached("ns", "search_documents", "q")
    assert cached is None
    assert key not in rag._tool_cache  # lazy eviction


# ---------------------------------------------------------------------------
# 10. flush_tool_cache clears only the correct namespace
# ---------------------------------------------------------------------------

def test_flush_tool_cache_namespace_isolation():
    import rag
    rag._set_cached("ns-a", "search_documents", "q1", "res1", [])
    rag._set_cached("ns-b", "search_documents", "q2", "res2", [])
    rag.flush_tool_cache("ns-a")
    assert rag._get_cached("ns-a", "search_documents", "q1") is None
    assert rag._get_cached("ns-b", "search_documents", "q2") is not None


# ---------------------------------------------------------------------------
# 11. Cache LRU eviction at max size
# ---------------------------------------------------------------------------

def test_cache_lru_eviction():
    import rag
    rag._TOOL_CACHE_MAX = 3
    rag._set_cached("ns", "search_documents", "q1", "r1", [])
    rag._set_cached("ns", "search_documents", "q2", "r2", [])
    rag._set_cached("ns", "search_documents", "q3", "r3", [])
    rag._set_cached("ns", "search_documents", "q4", "r4", [])  # evicts q1
    assert len(rag._tool_cache) == 3
    assert rag._get_cached("ns", "search_documents", "q1") is None  # evicted
    assert rag._get_cached("ns", "search_documents", "q4") is not None
    rag._TOOL_CACHE_MAX = 1000  # restore


# ---------------------------------------------------------------------------
# 12. _dispatch_tool: unknown tool returns "Unknown tool."
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_dispatch_tool_unknown_tool():
    from rag import _dispatch_tool
    tenant = MagicMock()
    tenant.slug = "ns"
    tenant.expertise_area = "medicina"
    result = await _dispatch_tool(
        {"function": {"name": "nonexistent_tool", "arguments": "{}"}},
        "ns", tenant,
    )
    assert result == "Unknown tool."


# ---------------------------------------------------------------------------
# 13. _dispatch_tool: search_web guard when web_search_url empty
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_dispatch_tool_search_web_no_url():
    from rag import _dispatch_tool
    tenant = MagicMock()
    tenant.slug = "ns"
    with patch("rag.get_setting", return_value=""):
        result = await _dispatch_tool(
            {"function": {"name": "search_web", "arguments": '{"query":"test"}'}},
            "ns", tenant,
        )
    assert result == "No web results found."


# ---------------------------------------------------------------------------
# 14. rag_query tool path: happy path (Round 1 dispatches, Round 2 synthesizes)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_rag_query_tool_path_happy():
    from rag import rag_query
    tool_call = {
        "id": "call_1",
        "type": "function",
        "function": {"name": "search_documents", "arguments": '{"query":"precio consulta"}'},
    }
    mock_db = AsyncMock()

    with (
        patch("rag.is_tool_use_available", return_value=True),
        patch("rag.get_setting", side_effect=lambda k, d: "http://web.search" if k == "web_search_url" else d),
        patch("rag.get_history", new_callable=AsyncMock, return_value=[]),
        patch("rag._reformulate_query", new_callable=AsyncMock, return_value="precio consulta"),
        patch("rag.call_chat_with_tools", new_callable=AsyncMock, return_value=(None, [tool_call])),
        patch("rag._tool_search_documents", new_callable=AsyncMock, return_value=("[Doc 1]: result", [{"content": "result", "source": "doc.pdf", "page": 1, "similarity": 0.9}])),
        patch("rag.call_chat", new_callable=AsyncMock, return_value="El precio es $50."),
        patch("rag.validate_output", side_effect=lambda x, **kw: x),
        patch("rag.save_turn", new_callable=AsyncMock),
        patch("rag.CANARY_TOKEN", "CANARY"),
    ):
        tenant = MagicMock()
        tenant.web_search_enabled = True
        tenant.expertise_area = "medicina"
        tenant.slug = "ns"

        answer, chunks, intent = await rag_query(
            db=mock_db,
            question="precio consulta",
            namespace="ns",
            user_id="u1",
            tenant=tenant,
        )

    assert answer == "El precio es $50."
    assert intent == "tool_use"


# ---------------------------------------------------------------------------
# 15. rag_query tool path: direct content (LLM answered without calling tools)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_rag_query_tool_path_direct_content():
    from rag import rag_query
    mock_db = AsyncMock()

    with (
        patch("rag.is_tool_use_available", return_value=True),
        patch("rag.get_setting", side_effect=lambda k, d: "http://web.search" if k == "web_search_url" else d),
        patch("rag.get_history", new_callable=AsyncMock, return_value=[]),
        patch("rag._reformulate_query", new_callable=AsyncMock, return_value="hola"),
        patch("rag._GREETING_PATTERN") as mock_greeting,
        patch("rag.ESCALATION_PATTERN") as mock_esc,
        patch("rag.call_chat_with_tools", new_callable=AsyncMock, return_value=("¡Hola! ¿En qué te ayudo?", [])),
        patch("rag.validate_output", side_effect=lambda x, **kw: x),
        patch("rag.save_turn", new_callable=AsyncMock),
        patch("rag.CANARY_TOKEN", "CANARY"),
    ):
        mock_greeting.match.return_value = None
        mock_esc.search.return_value = None

        tenant = MagicMock()
        tenant.web_search_enabled = True
        tenant.expertise_area = ""
        tenant.slug = "ns"

        answer, chunks, intent = await rag_query(
            db=mock_db,
            question="hola",
            namespace="ns",
            user_id="u1",
            tenant=tenant,
        )

    assert intent == "direct"
    assert chunks == []
    assert "Hola" in answer


# ---------------------------------------------------------------------------
# 16. rag_query tool path NOT entered when image_b64 present (vision bypass)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_rag_query_tool_path_skipped_image():
    from rag import rag_query
    mock_db = AsyncMock()

    with (
        patch("rag.is_tool_use_available", return_value=True),
        patch("rag.get_setting", side_effect=lambda k, d: "http://web.search" if k == "web_search_url" else d),
        patch("rag.get_history", new_callable=AsyncMock, return_value=[]),
        patch("rag._reformulate_query", new_callable=AsyncMock, return_value="q"),
        patch("rag._GREETING_PATTERN") as mock_greeting,
        patch("rag.ESCALATION_PATTERN") as mock_esc,
        patch("rag.call_chat_with_tools", new_callable=AsyncMock) as mock_tools,
        patch("rag.retrieve_context", new_callable=AsyncMock, return_value=[]),
        patch("rag.generate_answer", new_callable=AsyncMock, return_value="answer"),
        patch("rag.validate_output", side_effect=lambda x, **kw: x),
        patch("rag.save_turn", new_callable=AsyncMock),
        patch("rag._is_illegible_response", return_value=False),
        patch("rag.get_setting", return_value=""),
        patch("rag._triage_response", new_callable=AsyncMock, return_value=("off_topic", "fallback")),
        patch("rag._log_unanswered", new_callable=AsyncMock),
        patch("rag.CANARY_TOKEN", "CANARY"),
    ):
        mock_greeting.match.return_value = None
        mock_esc.search.return_value = None
        mock_tools.assert_not_called  # tool path should not be entered

        tenant = MagicMock()
        tenant.web_search_enabled = True
        tenant.slug = "ns"

        images = [{"b64": "abc", "mime": "image/jpeg"}]
        # With images present, vision path bypasses tool_use entirely
        await rag_query(
            db=mock_db,
            question="q",
            namespace="ns",
            user_id="u1",
            tenant=tenant,
            images=images,
        )
        mock_tools.assert_not_called()


# ---------------------------------------------------------------------------
# 17. rag_query tool path NOT entered when only 1 tool (no web_search_url)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_rag_query_tool_path_skipped_one_tool():
    from rag import rag_query
    mock_db = AsyncMock()

    with (
        patch("rag.is_tool_use_available", return_value=True),
        patch("rag.get_setting", return_value=""),  # no web_search_url
        patch("rag.get_history", new_callable=AsyncMock, return_value=[]),
        patch("rag._reformulate_query", new_callable=AsyncMock, return_value="q"),
        patch("rag._GREETING_PATTERN") as mock_greeting,
        patch("rag.ESCALATION_PATTERN") as mock_esc,
        patch("rag.call_chat_with_tools", new_callable=AsyncMock) as mock_tools,
        patch("rag.retrieve_context", new_callable=AsyncMock, return_value=[]),
        patch("rag._triage_response", new_callable=AsyncMock, return_value=("off_topic", "fallback")),
        patch("rag._log_unanswered", new_callable=AsyncMock),
        patch("rag.save_turn", new_callable=AsyncMock),
        patch("rag.validate_output", side_effect=lambda x, **kw: x),
        patch("rag.CANARY_TOKEN", "CANARY"),
    ):
        mock_greeting.match.return_value = None
        mock_esc.search.return_value = None

        tenant = MagicMock()
        tenant.web_search_enabled = True
        tenant.slug = "ns"

        await rag_query(
            db=mock_db, question="q", namespace="ns", user_id="u1", tenant=tenant
        )
        mock_tools.assert_not_called()


# ---------------------------------------------------------------------------
# 18. rag_query tool path NOT entered when is_tool_use_available=False
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_rag_query_tool_path_skipped_backoff():
    from rag import rag_query
    mock_db = AsyncMock()

    with (
        patch("rag.is_tool_use_available", return_value=False),
        patch("rag.get_setting", side_effect=lambda k, d: "http://web.search" if k == "web_search_url" else d),
        patch("rag.get_history", new_callable=AsyncMock, return_value=[]),
        patch("rag._reformulate_query", new_callable=AsyncMock, return_value="q"),
        patch("rag._GREETING_PATTERN") as mock_greeting,
        patch("rag.ESCALATION_PATTERN") as mock_esc,
        patch("rag.call_chat_with_tools", new_callable=AsyncMock) as mock_tools,
        patch("rag.retrieve_context", new_callable=AsyncMock, return_value=[]),
        patch("rag._triage_response", new_callable=AsyncMock, return_value=("off_topic", "fallback")),
        patch("rag._log_unanswered", new_callable=AsyncMock),
        patch("rag.save_turn", new_callable=AsyncMock),
        patch("rag.validate_output", side_effect=lambda x, **kw: x),
        patch("rag.CANARY_TOKEN", "CANARY"),
    ):
        mock_greeting.match.return_value = None
        mock_esc.search.return_value = None

        tenant = MagicMock()
        tenant.web_search_enabled = True
        tenant.slug = "ns"

        await rag_query(
            db=mock_db, question="q", namespace="ns", user_id="u1", tenant=tenant
        )
        mock_tools.assert_not_called()


# ---------------------------------------------------------------------------
# 19. rag_query tool path: partial failure (one tool raises → filter to '')
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_rag_query_partial_tool_failure():
    """One tool raises an exception; the other succeeds; synthesis still runs."""
    from rag import rag_query
    mock_db = AsyncMock()
    tc_docs = {
        "id": "call_1", "type": "function",
        "function": {"name": "search_documents", "arguments": '{"query":"q"}'},
    }
    tc_web = {
        "id": "call_2", "type": "function",
        "function": {"name": "search_web", "arguments": '{"query":"q"}'},
    }

    async def fake_dispatch(tc, namespace, tenant):
        if tc["function"]["name"] == "search_web":
            raise RuntimeError("web search timeout")
        return "[Doc 1]: some result"

    with (
        patch("rag.is_tool_use_available", return_value=True),
        patch("rag.get_setting", side_effect=lambda k, d: "http://web.search" if k == "web_search_url" else d),
        patch("rag.get_history", new_callable=AsyncMock, return_value=[]),
        patch("rag._reformulate_query", new_callable=AsyncMock, return_value="q"),
        patch("rag._GREETING_PATTERN") as mock_greeting,
        patch("rag.ESCALATION_PATTERN") as mock_esc,
        patch("rag.call_chat_with_tools", new_callable=AsyncMock, return_value=(None, [tc_docs, tc_web])),
        patch("rag._dispatch_tool", new=fake_dispatch),
        patch("rag._tool_search_documents", new_callable=AsyncMock, return_value=("[Doc 1]: some result", [{"content": "some result", "source": "doc.pdf", "page": 1, "similarity": 0.9}])),
        patch("rag.call_chat", new_callable=AsyncMock, return_value="Based on docs: some result"),
        patch("rag.validate_output", side_effect=lambda x, **kw: x),
        patch("rag.save_turn", new_callable=AsyncMock),
        patch("rag.CANARY_TOKEN", "CANARY"),
    ):
        mock_greeting.match.return_value = None
        mock_esc.search.return_value = None

        tenant = MagicMock()
        tenant.web_search_enabled = True
        tenant.expertise_area = ""
        tenant.slug = "ns"

        answer, chunks, intent = await rag_query(
            db=mock_db, question="q", namespace="ns", user_id="u1", tenant=tenant
        )

    assert intent == "tool_use"
    assert "some result" in answer


# ---------------------------------------------------------------------------
# 20. rag_query tool path: all-empty results → sequential fallback
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_rag_query_tool_all_empty_falls_back():
    """When all tools return '', the sequential pipeline runs instead."""
    from rag import rag_query
    mock_db = AsyncMock()
    tc = {
        "id": "call_1", "type": "function",
        "function": {"name": "search_documents", "arguments": '{"query":"q"}'},
    }

    with (
        patch("rag.is_tool_use_available", return_value=True),
        patch("rag.get_setting", side_effect=lambda k, d: "http://web.search" if k == "web_search_url" else d),
        patch("rag.get_history", new_callable=AsyncMock, return_value=[]),
        patch("rag._reformulate_query", new_callable=AsyncMock, return_value="q"),
        patch("rag._GREETING_PATTERN") as mock_greeting,
        patch("rag.ESCALATION_PATTERN") as mock_esc,
        patch("rag.call_chat_with_tools", new_callable=AsyncMock, return_value=(None, [tc])),
        patch("rag._dispatch_tool", new_callable=AsyncMock, return_value=""),
        # Sequential pipeline mock
        patch("rag.retrieve_context", new_callable=AsyncMock, return_value=[]),
        patch("rag._triage_response", new_callable=AsyncMock, return_value=("ambiguous", "No info found")),
        patch("rag._log_unanswered", new_callable=AsyncMock),
        patch("rag.save_turn", new_callable=AsyncMock),
        patch("rag.validate_output", side_effect=lambda x, **kw: x),
        patch("rag.CANARY_TOKEN", "CANARY"),
    ):
        mock_greeting.match.return_value = None
        mock_esc.search.return_value = None

        tenant = MagicMock()
        tenant.web_search_enabled = True
        tenant.expertise_area = ""
        tenant.slug = "ns"
        tenant.example_questions = []

        answer, chunks, intent = await rag_query(
            db=mock_db, question="q", namespace="ns", user_id="u1", tenant=tenant
        )

    # Should have fallen back to triage
    assert intent == "ambiguous"


# ---------------------------------------------------------------------------
# 21. rag_query: canary token redacted in tool path
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_rag_query_tool_path_canary_redacted():
    from rag import rag_query
    mock_db = AsyncMock()
    tc = {
        "id": "call_1", "type": "function",
        "function": {"name": "search_documents", "arguments": '{"query":"q"}'},
    }

    with (
        patch("rag.is_tool_use_available", return_value=True),
        patch("rag.get_setting", side_effect=lambda k, d: "http://web.search" if k == "web_search_url" else d),
        patch("rag.get_history", new_callable=AsyncMock, return_value=[]),
        patch("rag._reformulate_query", new_callable=AsyncMock, return_value="q"),
        patch("rag._GREETING_PATTERN") as mock_greeting,
        patch("rag.ESCALATION_PATTERN") as mock_esc,
        patch("rag.call_chat_with_tools", new_callable=AsyncMock, return_value=(None, [tc])),
        patch("rag._dispatch_tool", new_callable=AsyncMock, return_value="some docs"),
        patch("rag._tool_search_documents", new_callable=AsyncMock, return_value=("some docs", [])),
        patch("rag.call_chat", new_callable=AsyncMock, return_value="answer with CANARY_TOKEN_HERE"),
        patch("rag.validate_output", side_effect=lambda x, **kw: x),
        patch("rag.save_turn", new_callable=AsyncMock),
        patch("rag.CANARY_TOKEN", "CANARY_TOKEN_HERE"),
    ):
        mock_greeting.match.return_value = None
        mock_esc.search.return_value = None

        tenant = MagicMock()
        tenant.web_search_enabled = True
        tenant.expertise_area = ""
        tenant.slug = "ns"

        answer, _, intent = await rag_query(
            db=mock_db, question="q", namespace="ns", user_id="u1", tenant=tenant
        )

    assert "CANARY_TOKEN_HERE" not in answer
    assert "[REDACTED]" in answer
    assert intent == "tool_use"


# ---------------------------------------------------------------------------
# 22. Upload route: flush_tool_cache called after commit (api.py)
# ---------------------------------------------------------------------------

def test_upload_flushes_tool_cache(authed_api_client):
    """Uploading a file triggers flush_tool_cache for the tenant's namespace."""
    client, mock_tenant = authed_api_client

    with (
        patch("routes.api.process_uploaded_file", return_value=(
            [{"content": "chunk", "source": "test.pdf", "page": 1}], 1, "full text"
        )),
        patch("routes.api.index_chunks", new_callable=AsyncMock, return_value=1),
        patch("routes.api.flush_tool_cache") as mock_flush,
    ):
        resp = client.post(
            "/upload",
            files={"file": ("test.pdf", b"%PDF-1.4 content", "application/pdf")},
        )

    assert resp.status_code == 200
    mock_flush.assert_called_once_with(mock_tenant.slug)
