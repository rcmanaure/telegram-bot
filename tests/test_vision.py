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


# ─── rag.py: generate_answer with image(s) ────────────────────────────────────

@pytest.mark.asyncio
async def test_generate_answer_with_image_builds_content_array():
    """When images is set, last message content is a list with text + image_url."""
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
                images=[{"b64": "abc123", "mime": "image/jpeg"}],
            )

    user_msg = captured_messages[-1]
    assert isinstance(user_msg["content"], list)
    types = [part["type"] for part in user_msg["content"]]
    assert "text" in types
    assert "image_url" in types


@pytest.mark.asyncio
async def test_generate_answer_with_multiple_images_builds_content_array():
    """Multiple images produce multiple image_url parts in content array."""
    chunks = [{"content": "info", "source": "doc.pdf", "page": 1}]
    captured = []

    async def mock_call_chat(messages, **kwargs):
        captured.extend(messages)
        return "answer"

    with patch("rag.call_chat", side_effect=mock_call_chat):
        with patch("rag.settings") as mock_settings:
            mock_settings.llm_vision_model = "llava"
            from rag import generate_answer
            await generate_answer(
                chunks,
                "¿Qué dicen estas imágenes?",
                [],
                images=[
                    {"b64": "img1", "mime": "image/jpeg"},
                    {"b64": "img2", "mime": "image/png"},
                ],
            )

    user_msg = captured[-1]
    assert isinstance(user_msg["content"], list)
    img_parts = [p for p in user_msg["content"] if p["type"] == "image_url"]
    assert len(img_parts) == 2
    assert img_parts[0]["image_url"]["url"] == "data:image/jpeg;base64,img1"
    assert img_parts[1]["image_url"]["url"] == "data:image/png;base64,img2"


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
            await generate_answer(chunks, "q", [], images=[{"b64": "AAAA", "mime": "image/png"}])

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


# ─── bot.py: handle_photo (with image buffer) ────────────────────────────────

def _make_photo_update(user_id=123, caption=None, file_size=100_000, media_group_id=None):
    from telegram import Update
    update = MagicMock(spec=Update)
    update.effective_user = MagicMock()
    update.effective_user.id = user_id
    update.effective_user.language_code = "es"
    update.effective_chat = MagicMock()
    update.effective_chat.id = 999
    update.message = MagicMock()
    update.message.caption = caption
    update.message.media_group_id = media_group_id
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
async def test_handle_photo_adds_to_buffer():
    """Photo handler adds image to buffer instead of calling _process_question directly."""
    update = _make_photo_update(caption="¿cuánto cuesta?")
    ctx = _make_photo_ctx()

    fake_file = MagicMock()
    fake_file.download_as_bytearray = AsyncMock(return_value=bytearray(b"fake"))
    ctx.bot.get_file = AsyncMock(return_value=fake_file)

    with patch("bot.image_buffer") as mock_buffer, \
         patch("bot.sanitize_user_input", side_effect=lambda x: x):
        mock_buffer.add_image = MagicMock(return_value=None)  # synchronous mock
        from bot import handle_photo
        await handle_photo(update, ctx)

    # Buffer was called with the image
    mock_buffer.add_image.assert_called_once()
    call_kwargs = mock_buffer.add_image.call_args[1]
    assert call_kwargs["b64"] is not None  # base64 of image
    assert call_kwargs["mime"] is not None
    assert call_kwargs["question"] == "¿cuánto cuesta?"


@pytest.mark.asyncio
async def test_handle_photo_album_uses_media_group_id():
    """Photos with media_group_id use album-specific buffer key."""
    update = _make_photo_update(caption=None, media_group_id="album_123")
    ctx = _make_photo_ctx()

    fake_file = MagicMock()
    fake_file.download_as_bytearray = AsyncMock(return_value=bytearray(b"fake"))
    ctx.bot.get_file = AsyncMock(return_value=fake_file)

    with patch("bot.image_buffer") as mock_buffer, \
         patch("bot.sanitize_user_input", side_effect=lambda x: x):
        mock_buffer.add_image = MagicMock(return_value=None)
        from bot import handle_photo
        await handle_photo(update, ctx)

    call_kwargs = mock_buffer.add_image.call_args[1]
    assert "album_123" in call_kwargs["key"]


@pytest.mark.asyncio
async def test_handle_photo_single_uses_pending_key():
    """Photo without media_group_id uses _pending_ buffer key."""
    update = _make_photo_update(caption=None, media_group_id=None)
    ctx = _make_photo_ctx()

    fake_file = MagicMock()
    fake_file.download_as_bytearray = AsyncMock(return_value=bytearray(b"fake"))
    ctx.bot.get_file = AsyncMock(return_value=fake_file)

    with patch("bot.image_buffer") as mock_buffer, \
         patch("bot.sanitize_user_input", side_effect=lambda x: x):
        mock_buffer.add_image = MagicMock(return_value=None)
        from bot import handle_photo
        await handle_photo(update, ctx)

    call_kwargs = mock_buffer.add_image.call_args[1]
    assert "_pending_" in call_kwargs["key"]


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