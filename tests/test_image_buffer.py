"""
Tests for the image buffer module (multi-image debounce + flush).
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from image_buffer import ImageBuffer, BufferedImage, MAX_IMAGES, BUFFER_WINDOW_SECS, MAX_ENTRY_TTL


# ─── Single image flush ───────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_single_image_flushes_after_window():
    """A single image is flushed after the buffer window expires."""
    buf = ImageBuffer()
    flushed = []

    async def on_flush(images_dicts, question):
        flushed.append((images_dicts, question))

    buf.add_image("k1", b64="abc", mime="image/jpeg", question="¿Qué es?", on_flush=on_flush, buffer_window=0.05)

    # Wait just past the buffer window
    await asyncio.sleep(0.15)

    assert len(flushed) == 1
    assert flushed[0][0] == [{"b64": "abc", "mime": "image/jpeg"}]
    assert flushed[0][1] == "¿Qué es?"
    buf.clear()


# ─── Multiple images debounce ─────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_multiple_images_debounce_into_single_flush():
    """Two images added within the window flush together as one batch."""
    buf = ImageBuffer()
    flushed = []

    async def on_flush(images_dicts, question):
        flushed.append((images_dicts, question))

    # Add first image with very short window
    buf.add_image("k1", b64="img1", mime="image/jpeg", question="", on_flush=on_flush, buffer_window=0.1)
    # Add second image quickly (within window) — resets timer
    buf.add_image("k1", b64="img2", mime="image/png", question="¿Qué son?", on_flush=on_flush, buffer_window=0.1)

    # Wait for flush
    await asyncio.sleep(0.3)

    assert len(flushed) == 1
    images, question = flushed[0]
    assert len(images) == 2
    assert images[0]["b64"] == "img1"
    assert images[1]["b64"] == "img2"
    # Last non-empty question wins
    assert question == "¿Qué son?"
    buf.clear()


# ─── Max images rejection ──────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_max_images_rejects_sixth():
    """Adding a 6th image returns an error string; first 5 are kept."""
    buf = ImageBuffer()
    flushed = []

    async def on_flush(images_dicts, question):
        flushed.append((images_dicts, question))

    for i in range(5):
        buf.add_image("k1", b64=f"img{i}", mime="image/jpeg", question="", on_flush=on_flush, buffer_window=0.5)

    # 6th image should be rejected
    error = buf.add_image("k1", b64="img5", mime="image/jpeg", question="", on_flush=on_flush, buffer_window=0.5)
    assert error is not None
    assert str(MAX_IMAGES) in error

    # Entry should still have exactly 5 images
    entry = buf.get_entry("k1")
    assert entry is not None
    assert len(entry.images) == MAX_IMAGES
    buf.clear()


# ─── Different keys are independent ────────────────────────────────────────────

@pytest.mark.asyncio
async def test_different_keys_independent():
    """Images under different buffer keys flush independently."""
    buf = ImageBuffer()
    flushed_k1 = []
    flushed_k2 = []

    async def on_flush_k1(images_dicts, question):
        flushed_k1.append((images_dicts, question))

    async def on_flush_k2(images_dicts, question):
        flushed_k2.append((images_dicts, question))

    buf.add_image("k1", b64="a1", mime="image/jpeg", question="q1", on_flush=on_flush_k1, buffer_window=0.05)
    buf.add_image("k2", b64="b1", mime="image/png", question="q2", on_flush=on_flush_k2, buffer_window=0.05)

    await asyncio.sleep(0.15)

    assert len(flushed_k1) == 1
    assert len(flushed_k2) == 1
    assert flushed_k1[0][0] == [{"b64": "a1", "mime": "image/jpeg"}]
    assert flushed_k2[0][0] == [{"b64": "b1", "mime": "image/png"}]
    buf.clear()


# ─── Sweep removes stale, preserves fresh ──────────────────────────────────────

@pytest.mark.asyncio
async def test_sweep_removes_stale_preserves_fresh():
    """sweep() removes entries older than MAX_ENTRY_TTL."""
    import time
    buf = ImageBuffer()

    # Manually insert a stale entry
    from image_buffer import _BufferEntry
    stale_entry = _BufferEntry(created_at=time.monotonic() - MAX_ENTRY_TTL - 10)
    buf._entries["stale"] = stale_entry

    # Insert a fresh entry (needs running event loop for create_task)
    async def noop(images, question):
        pass
    buf.add_image("fresh", b64="x", mime="image/jpeg", question="q", on_flush=noop, buffer_window=60)

    removed = buf.sweep()
    assert removed == 1
    assert "stale" not in buf._entries
    assert "fresh" in buf._entries
    buf.clear()


# ─── Cancel on debounce ────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_debounce_cancels_previous_timer():
    """Adding a second image cancels the first timer — only one flush happens."""
    buf = ImageBuffer()
    flushed = []

    async def on_flush(images_dicts, question):
        flushed.append((images_dicts, question))

    # First image: flush after 0.1s
    buf.add_image("k1", b64="img1", mime="image/jpeg", question="", on_flush=on_flush, buffer_window=0.1)
    # After 0.05s (before flush), add second image with longer window
    await asyncio.sleep(0.05)
    buf.add_image("k1", b64="img2", mime="image/jpeg", question="q2", on_flush=on_flush, buffer_window=0.1)

    # At 0.2s total, the first timer would have fired but was cancelled
    await asyncio.sleep(0.2)

    # Only one flush (the debounce-rescheduled one)
    assert len(flushed) == 1
    assert len(flushed[0][0]) == 2  # both images
    buf.clear()


# ─── Flush pops entry ──────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_flush_pops_entry():
    """After flush, the entry is removed from the buffer."""
    buf = ImageBuffer()
    flushed = []

    async def on_flush(images_dicts, question):
        flushed.append((images_dicts, question))

    buf.add_image("k1", b64="x", mime="image/jpeg", question="q", on_flush=on_flush, buffer_window=0.05)
    assert buf.get_entry("k1") is not None

    await asyncio.sleep(0.15)

    assert buf.get_entry("k1") is None  # popped on flush
    buf.clear()


# ─── Flush exception doesn't leak entry ────────────────────────────────────────

@pytest.mark.asyncio
async def test_flush_exception_does_not_leak_entry():
    """If on_flush raises, the entry is still removed from the buffer."""
    buf = ImageBuffer()

    async def bad_flush(images_dicts, question):
        raise RuntimeError("boom")

    buf.add_image("k1", b64="x", mime="image/jpeg", question="q", on_flush=bad_flush, buffer_window=0.05)
    await asyncio.sleep(0.15)

    # Entry must be popped even on exception
    assert buf.get_entry("k1") is None
    buf.clear()


# ─── Question last-non-empty wins ──────────────────────────────────────────────

@pytest.mark.asyncio
async def test_question_last_non_empty_wins():
    """Last non-empty question overwrites previous empty/blank questions."""
    buf = ImageBuffer()
    flushed = []

    async def on_flush(images_dicts, question):
        flushed.append((images_dicts, question))

    # First image with no caption
    buf.add_image("k1", b64="i1", mime="image/jpeg", question="", on_flush=on_flush, buffer_window=0.1)
    # Second image with caption
    buf.add_image("k1", b64="i2", mime="image/jpeg", question="¿Cuánto cuesta?", on_flush=on_flush, buffer_window=0.1)
    # Third image with no caption — previous non-empty question preserved
    buf.add_image("k1", b64="i3", mime="image/jpeg", question="", on_flush=on_flush, buffer_window=0.1)

    await asyncio.sleep(0.3)

    assert len(flushed) == 1
    assert flushed[0][1] == "¿Cuánto cuesta?"
    buf.clear()


# ─── Default question for single vs multiple images ────────────────────────────

@pytest.mark.asyncio
async def test_default_question_singular_for_single_image():
    """When no question provided and only 1 image, uses singular default."""
    buf = ImageBuffer()
    flushed = []

    async def on_flush(images_dicts, question):
        flushed.append((images_dicts, question))

    buf.add_image("k1", b64="x", mime="image/jpeg", question="", on_flush=on_flush, buffer_window=0.05)
    await asyncio.sleep(0.15)

    assert "esta imagen" in flushed[0][1]
    buf.clear()


@pytest.mark.asyncio
async def test_default_question_plural_for_multiple_images():
    """When no question provided and multiple images, uses plural default."""
    buf = ImageBuffer()
    flushed = []

    async def on_flush(images_dicts, question):
        flushed.append((images_dicts, question))

    buf.add_image("k1", b64="i1", mime="image/jpeg", question="", on_flush=on_flush, buffer_window=0.1)
    buf.add_image("k1", b64="i2", mime="image/jpeg", question="", on_flush=on_flush, buffer_window=0.1)
    await asyncio.sleep(0.3)

    assert "estas imágenes" in flushed[0][1]
    buf.clear()


# ─── Clear cancels tasks ───────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_clear_cancels_pending_tasks():
    """clear() cancels all pending flush tasks and removes entries."""
    buf = ImageBuffer()
    flushed = []

    async def on_flush(images_dicts, question):
        flushed.append((images_dicts, question))

    buf.add_image("k1", b64="x", mime="image/jpeg", question="q", on_flush=on_flush, buffer_window=60)
    assert buf.pending_count() == 1

    buf.clear()
    assert buf.pending_count() == 0

    # Wait and confirm no flush happened
    await asyncio.sleep(0.1)
    assert len(flushed) == 0


# ─── flush_by_prefix (text/voice triggers image flush) ─────────────────────────

@pytest.mark.asyncio
async def test_flush_by_prefix_flushes_matching_keys():
    """flush_by_prefix flushes all entries whose key starts with the prefix."""
    buf = ImageBuffer()
    flushed = []

    async def on_flush(images_dicts, question):
        flushed.append((images_dicts, question))

    # Two entries under the same user prefix
    buf.add_image("tg:123:album1", b64="a1", mime="image/jpeg", question="", on_flush=on_flush, buffer_window=60)
    buf.add_image("tg:123:_pending_", b64="b1", mime="image/jpeg", question="", on_flush=on_flush, buffer_window=60)
    # Unrelated user — should NOT be flushed
    buf.add_image("tg:456:_pending_", b64="c1", mime="image/jpeg", question="", on_flush=on_flush, buffer_window=60)

    count = await buf.flush_by_prefix("tg:123:", override_question="¿Cuánto cuesta?")
    assert count == 2
    assert len(flushed) == 2

    # Unrelated user entry still in buffer
    assert buf.get_entry("tg:456:_pending_") is not None
    buf.clear()


@pytest.mark.asyncio
async def test_flush_by_prefix_override_question_when_empty():
    """flush_by_prefix sets override_question on entries that have no question."""
    buf = ImageBuffer()
    flushed = []

    async def on_flush(images_dicts, question):
        flushed.append((images_dicts, question))

    # Image with NO caption → empty question
    buf.add_image("tg:99:_pending_", b64="x", mime="image/jpeg", question="", on_flush=on_flush, buffer_window=60)

    await buf.flush_by_prefix("tg:99:", override_question="¿Qué dice la orden?")
    assert len(flushed) == 1
    assert flushed[0][1] == "¿Qué dice la orden?"
    buf.clear()


@pytest.mark.asyncio
async def test_flush_by_prefix_keeps_existing_question():
    """flush_by_prefix does NOT override if the entry already has a question."""
    buf = ImageBuffer()
    flushed = []

    async def on_flush(images_dicts, question):
        flushed.append((images_dicts, question))

    # Image WITH caption
    buf.add_image("tg:99:_pending_", b64="x", mime="image/jpeg", question="¿Cuánto cuesta la biopsia?", on_flush=on_flush, buffer_window=60)

    await buf.flush_by_prefix("tg:99:", override_question="¿Qué es esto?")
    assert len(flushed) == 1
    # Original caption question kept, not overridden
    assert flushed[0][1] == "¿Cuánto cuesta la biopsia?"
    buf.clear()


@pytest.mark.asyncio
async def test_flush_by_prefix_no_matching_keys():
    """flush_by_prefix returns 0 when no keys match."""
    buf = ImageBuffer()

    async def noop(images, question):
        pass
    buf.add_image("tg:123:_pending_", b64="x", mime="image/jpeg", question="q", on_flush=noop, buffer_window=60)

    count = await buf.flush_by_prefix("tg:999:")
    assert count == 0
    # Original entry untouched
    assert buf.get_entry("tg:123:_pending_") is not None
    buf.clear()