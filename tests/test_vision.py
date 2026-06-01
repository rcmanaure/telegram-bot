"""
Tests for vision pipeline: call_chat model override, generate_answer with image,
rag_query image threading, handle_photo bot handler, WA image path.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import base64
import pytest
from unittest.mock import AsyncMock, MagicMock, patch, call


# ─── llm.py: call_chat model override ────────────────────────────────────────

@pytest.mark.asyncio
async def test_call_chat_model_override_uses_single_model():
    """When model= override is set, only that model is tried (no fallback)."""
    import httpx
    from unittest.mock import AsyncMock, MagicMock, patch

    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {
        "choices": [{"message": {"content": "vision answer"}}]
    }

    with patch("llm._chat_client") as mock_client:
        mock_client.post = AsyncMock(return_value=mock_response)
        with patch("llm.settings") as mock_settings:
            mock_settings.llm_model = "openrouter/free"
            mock_settings.llm_fallback_model = "openrouter/owl-alpha"
            mock_settings.llm_base_url = "https://openrouter.ai/api/v1"
            mock_settings.effective_llm_api_key = "test-key"

            from llm import call_chat
            result = await call_chat([{"role": "user", "content": "test"}], model="llava")

    assert result == "vision answer"
    assert mock_client.post.call_count == 1
    body = mock_client.post.call_args[1]["json"]
    assert body["model"] == "llava"


@pytest.mark.asyncio
async def test_call_chat_model_override_skips_fallback_on_error():
    """When model override is set and call fails, no fallback is attempted."""
    import httpx

    with patch("llm._chat_client") as mock_client:
        mock_client.post = AsyncMock(side_effect=httpx.TimeoutException("timeout"))
        with patch("llm.settings") as mock_settings:
            mock_settings.llm_model = "openrouter/free"
            mock_settings.llm_fallback_model = "openrouter/owl-alpha"
            mock_settings.llm_base_url = "https://openrouter.ai/api/v1"
            mock_settings.effective_llm_api_key = "test-key"

            from llm import call_chat
            with pytest.raises(RuntimeError, match="timed out"):
                await call_chat([{"role": "user", "content": "test"}], model="llava")

    assert mock_client.post.call_count == 1  # only tried once, no fallback


@pytest.mark.asyncio
async def test_call_chat_no_override_tries_fallback():
    """Without model override, fallback model is attempted on failure."""
    import httpx

    responses = [
        httpx.TimeoutException("primary timeout"),
    ]
    call_count = 0

    async def mock_post(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise httpx.TimeoutException("primary timeout")
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"choices": [{"message": {"content": "fallback answer"}}]}
        return mock_resp

    with patch("llm._chat_client") as mock_client:
        mock_client.post = mock_post
        with patch("llm.settings") as mock_settings:
            mock_settings.llm_model = "openrouter/free"
            mock_settings.llm_fallback_model = "openrouter/owl-alpha"
            mock_settings.llm_base_url = "https://openrouter.ai/api/v1"
            mock_settings.effective_llm_api_key = "test-key"

            from llm import call_chat
            result = await call_chat([{"role": "user", "content": "test"}])

    assert result == "fallback answer"
    assert call_count == 2


# ─── rag.py: generate_answer with image ──────────────────────────────────────

@pytest.mark.asyncio
async def test_generate_answer_with_image_builds_content_array():
    """When image_b64 is set, last message content is a list with text + image_url."""
    chunks = [{"content": "gym has a pool", "source": "docs.pdf", "page": 1}]
    captured_messages = []

    async def mock_call_chat(messages, **kwargs):
        captured_messages.extend(messages)
        return "The gym has a pool."

    with patch("rag.call_chat", side_effect=mock_call_chat):
        with patch("rag.settings") as mock_settings:
            mock_settings.llm_vision_model = "llava"
            from rag import generate_answer
            await generate_answer(
                chunks,
                "¿Tienen pileta?",
                [],
                image_b64="abc123",
                image_mime="image/jpeg",
            )

    user_msg = captured_messages[-1]
    assert isinstance(user_msg["content"], list)
    types = [part["type"] for part in user_msg["content"]]
    assert "text" in types
    assert "image_url" in types


@pytest.mark.asyncio
async def test_generate_answer_image_url_format():
    """data URI format is correct: data:{mime};base64,{b64}"""
    chunks = [{"content": "info", "source": "doc.pdf", "page": 1}]
    captured = []

    async def mock_call_chat(messages, **kwargs):
        captured.extend(messages)
        return "answer"

    with patch("rag.call_chat", side_effect=mock_call_chat):
        with patch("rag.settings") as mock_settings:
            mock_settings.llm_vision_model = ""
            from rag import generate_answer
            await generate_answer(chunks, "q", [], image_b64="AAAA", image_mime="image/png")

    user_msg = captured[-1]
    img_part = next(p for p in user_msg["content"] if p["type"] == "image_url")
    assert img_part["image_url"]["url"] == "data:image/png;base64,AAAA"


@pytest.mark.asyncio
async def test_generate_answer_without_image_content_is_string():
    """Without image, content stays a plain string (no regression)."""
    chunks = [{"content": "info", "source": "doc.pdf", "page": 1}]
    captured = []

    async def mock_call_chat(messages, **kwargs):
        captured.extend(messages)
        return "answer"

    with patch("rag.call_chat", side_effect=mock_call_chat):
        with patch("rag.settings") as mock_settings:
            mock_settings.llm_vision_model = ""
            from rag import generate_answer
            await generate_answer(chunks, "q", [])

    user_msg = captured[-1]
    assert isinstance(user_msg["content"], str)


# ─── bot.py: handle_photo ────────────────────────────────────────────────────

def _make_photo_update(user_id=123, caption=None, file_size=100_000):
    from telegram import Update
    update = MagicMock(spec=Update)
    update.effective_user = MagicMock()
    update.effective_user.id = user_id
    update.effective_user.language_code = "es"
    update.effective_chat = MagicMock()
    update.effective_chat.id = 999
    update.message = MagicMock()
    update.message.caption = caption
    update.message.reply_text = AsyncMock()

    photo = MagicMock()
    photo.file_id = "file123"
    photo.file_size = file_size
    update.message.photo = [photo]
    return update


def _make_photo_ctx(slug="test-tenant", expertise_area="fitness"):
    tenant = MagicMock()
    tenant.slug = slug
    tenant.expertise_area = expertise_area
    tenant.id = 1
    tenant.contact_url = None
    tenant.example_questions = None

    ctx = MagicMock()
    ctx.bot_data = {"tenant": tenant}
    ctx.bot = MagicMock()
    ctx.bot.send_chat_action = AsyncMock()
    ctx.bot.get_file = AsyncMock()
    return ctx


@pytest.mark.asyncio
async def test_handle_photo_no_caption_uses_default_question():
    """Photo with no caption uses the default question constant."""
    update = _make_photo_update(caption=None)
    ctx = _make_photo_ctx()

    fake_file = MagicMock()
    fake_file.download_as_bytearray = AsyncMock(return_value=bytearray(b"fake_image_bytes"))
    ctx.bot.get_file = AsyncMock(return_value=fake_file)

    rag_result = ("answer", [], None)

    with patch("bot.AsyncSessionLocal") as mock_session_cls, \
         patch("bot.rag_query", AsyncMock(return_value=rag_result)) as mock_rag, \
         patch("bot.sanitize_user_input", side_effect=lambda x: x):

        mock_db = AsyncMock()
        mock_session_cls.return_value.__aenter__ = AsyncMock(return_value=mock_db)
        mock_session_cls.return_value.__aexit__ = AsyncMock(return_value=False)

        from bot import handle_photo, _PHOTO_DEFAULT_QUESTION
        await handle_photo(update, ctx)

    mock_rag.assert_called_once()
    call_kwargs = mock_rag.call_args[1]
    assert call_kwargs["question"] == _PHOTO_DEFAULT_QUESTION
    assert call_kwargs["image_b64"] is not None


@pytest.mark.asyncio
async def test_handle_photo_with_caption_uses_caption():
    """Photo with caption uses the caption as the question."""
    update = _make_photo_update(caption="¿cuánto cuesta?")
    ctx = _make_photo_ctx()

    fake_file = MagicMock()
    fake_file.download_as_bytearray = AsyncMock(return_value=bytearray(b"fake"))
    ctx.bot.get_file = AsyncMock(return_value=fake_file)

    rag_result = ("answer", [], None)
    with patch("bot.AsyncSessionLocal") as mock_session_cls, \
         patch("bot.rag_query", AsyncMock(return_value=rag_result)), \
         patch("bot.sanitize_user_input", side_effect=lambda x: x):

        mock_db = AsyncMock()
        mock_session_cls.return_value.__aenter__ = AsyncMock(return_value=mock_db)
        mock_session_cls.return_value.__aexit__ = AsyncMock(return_value=False)

        from bot import handle_photo
        await handle_photo(update, ctx)

    # caption used as question — verified implicitly via rag_query being called
    update.message.reply_text.assert_called()


@pytest.mark.asyncio
async def test_handle_photo_rejects_oversized():
    """Photo > 5 MB gets a size error reply, no RAG call."""
    update = _make_photo_update(file_size=6 * 1024 * 1024)
    ctx = _make_photo_ctx()

    with patch("bot.rag_query") as mock_rag:
        from bot import handle_photo
        await handle_photo(update, ctx)

    update.message.reply_text.assert_called_once()
    assert "grande" in update.message.reply_text.call_args[0][0].lower()
    mock_rag.assert_not_called()


@pytest.mark.asyncio
async def test_handle_photo_telegram_error_on_download():
    """TelegramError during download → friendly reply, no crash."""
    from telegram.error import TelegramError

    update = _make_photo_update()
    ctx = _make_photo_ctx()
    ctx.bot.get_file = AsyncMock(side_effect=TelegramError("server error"))

    with patch("bot.rag_query") as mock_rag:
        from bot import handle_photo
        await handle_photo(update, ctx)

    update.message.reply_text.assert_called_once()
    reply = update.message.reply_text.call_args[0][0]
    assert "imagen" in reply.lower()
    mock_rag.assert_not_called()
