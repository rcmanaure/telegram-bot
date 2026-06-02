"""Image buffer — collects multiple images before RAG processing.

Single-worker only (1 uvicorn process). Uses asyncio.create_task with
debounce to batch images arriving in quick succession.

When a user sends multiple photos (e.g., a Telegram album or rapid WA
messages), each image is added to the buffer and a flush timer is set.
If another image arrives before the timer fires, the timer resets.
When the timer finally fires, all collected images are processed together
in a single RAG query with multiple image_url parts in the vision API call.
"""
import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Awaitable, Callable

logger = logging.getLogger(__name__)

BUFFER_WINDOW_SECS = 2.5
MAX_IMAGES = 5
MAX_ENTRY_TTL = 30  # safety net: entries older than this are swept


@dataclass
class BufferedImage:
    """A single base64-encoded image awaiting processing."""
    b64: str
    mime: str


@dataclass
class _BufferEntry:
    """Internal buffer state for one grouping key."""
    images: list[BufferedImage] = field(default_factory=list)
    question: str = ""
    on_flush: Callable[[list[dict], str], Awaitable[None]] | None = None
    flush_task: asyncio.Task | None = None
    created_at: float = 0.0
    buffer_window: float = BUFFER_WINDOW_SECS


class ImageBuffer:
    """In-memory image buffer with debounce flush.

    add_image() appends an image and schedules/resets a timer.
    When the timer fires, on_flush(images_dicts, question) is called.

    Buffer keys follow the pattern:
      - TG album:  "tg:{user_id}:{media_group_id}"
      - TG single: "tg:{user_id}:_pending_"
      - WA:        "wa:{tenant_slug}:{user_id}"
    """

    def __init__(self):
        self._entries: dict[str, _BufferEntry] = {}

    def add_image(
        self,
        key: str,
        b64: str,
        mime: str,
        question: str,
        on_flush: Callable[[list[dict], str], Awaitable[None]],
        buffer_window: float | None = None,
    ) -> str | None:
        """Add an image to the buffer.

        Returns None on success, or an error message string if max images exceeded.
        """
        now = time.monotonic()

        if key not in self._entries:
            self._entries[key] = _BufferEntry(
                created_at=now,
                buffer_window=buffer_window or BUFFER_WINDOW_SECS,
            )

        entry = self._entries[key]

        if len(entry.images) >= MAX_IMAGES:
            logger.warning("image_buffer_max key=%s count=%d", key, len(entry.images))
            return f"No puedo procesar más de {MAX_IMAGES} imágenes a la vez. " \
                   f"Las primeras {MAX_IMAGES} serán procesadas."

        entry.images.append(BufferedImage(b64=b64, mime=mime))
        if question.strip():
            entry.question = question  # last non-empty caption wins
        entry.on_flush = on_flush  # always update (caller may have fresh context)

        # Cancel existing flush task and schedule a new one (debounce)
        if entry.flush_task and not entry.flush_task.done():
            entry.flush_task.cancel()
        entry.flush_task = asyncio.create_task(self._flush_after(key, entry.buffer_window))

        logger.info(
            "image_buffer_add key=%s count=%d window=%.1fs",
            key, len(entry.images), entry.buffer_window,
        )
        return None

    async def _flush_after(self, key: str, delay: float) -> None:
        """Sleep for delay, then flush the buffer entry."""
        try:
            await asyncio.sleep(delay)
        except asyncio.CancelledError:
            return  # debounced away — a newer image reset the timer
        await self._flush(key)

    async def _flush(self, key: str) -> None:
        """Process buffered images for the given key."""
        entry = self._entries.pop(key, None)
        if entry is None:
            return
        # Cancel the debounce timer — but NOT if we're already running inside it.
        # When _flush_after calls _flush, entry.flush_task IS the current task;
        # cancelling it would raise CancelledError at the next await (on_flush),
        # silently killing the callback. Only cancel when called from a
        # different context (e.g. flush_by_prefix from a text/voice handler).
        current_task = asyncio.current_task()
        if entry.flush_task and entry.flush_task is not current_task and not entry.flush_task.done():
            entry.flush_task.cancel()

        images_dicts = [{"b64": img.b64, "mime": img.mime} for img in entry.images]
        # Use singular/plural default question
        if not entry.question.strip():
            if len(entry.images) == 1:
                entry.question = "¿Qué querés saber sobre esta imagen?"
            else:
                entry.question = "¿Qué querés saber sobre estas imágenes?"

        logger.info(
            "image_buffer_flush key=%s images=%d question=%r",
            key, len(images_dicts), entry.question[:60],
        )

        if entry.on_flush:
            try:
                await entry.on_flush(images_dicts, entry.question)
            except Exception:
                logger.exception("image_buffer_flush_error key=%s", key)

    def get_entry(self, key: str) -> _BufferEntry | None:
        """For testing: inspect a buffer entry."""
        return self._entries.get(key)

    def pending_count(self) -> int:
        """For testing: number of pending buffer entries."""
        return len(self._entries)

    async def flush_by_prefix(self, prefix: str, override_question: str = "") -> int:
        """Flush all entries whose key starts with prefix.

        Used when a text/voice message arrives while images are buffered:
        the images get processed immediately with the text as their question.

        If override_question is provided and the entry has no question,
        the override is used. If the entry already has a question, it's kept.

        Returns count of entries flushed.
        """
        keys = [k for k in self._entries if k.startswith(prefix)]
        for key in keys:
            entry = self._entries.get(key)
            if entry and override_question.strip() and not entry.question.strip():
                entry.question = override_question
        # Flush each entry (pops from dict)
        for key in keys:
            await self._flush(key)
        return len(keys)

    def sweep(self) -> int:
        """Remove entries older than MAX_ENTRY_TTL. Returns count removed."""
        now = time.monotonic()
        cutoff = now - MAX_ENTRY_TTL
        stale = [
            k for k, v in self._entries.items()
            if v.created_at <= cutoff
        ]
        for k in stale:
            entry = self._entries.pop(k)
            if entry.flush_task and not entry.flush_task.done():
                entry.flush_task.cancel()
        if stale:
            logger.debug("image_buffer_sweep removed=%d remaining=%d", len(stale), len(self._entries))
        return len(stale)

    def clear(self) -> None:
        """For testing: clear all entries and cancel all tasks."""
        for entry in self._entries.values():
            if entry.flush_task and not entry.flush_task.done():
                entry.flush_task.cancel()
        self._entries.clear()


# Module-level instance (single worker)
image_buffer = ImageBuffer()