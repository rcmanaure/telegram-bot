"""Speech-to-text — Groq Whisper transcription."""
import logging

import httpx

from config import settings

logger = logging.getLogger(__name__)

http_client = httpx.AsyncClient(timeout=60)


async def transcribe_voice(audio_bytes: bytes, filename: str = "voice.ogg") -> str:
    """Transcribe audio bytes using Groq Whisper. Raises RuntimeError on failure."""
    from config_overlay import get_setting
    groq_key = get_setting("groq_api_key", settings.groq_api_key)
    if not groq_key:
        raise RuntimeError("GROQ_API_KEY not configured.")
    try:
        response = await http_client.post(
            "https://api.groq.com/openai/v1/audio/transcriptions",
            headers={"Authorization": f"Bearer {groq_key}"},
            files={"file": (filename, audio_bytes, "audio/ogg")},
            data={"model": "whisper-large-v3-turbo", "response_format": "text"},
        )
        response.raise_for_status()
        result = response.text.strip()
        logger.debug("transcribe_voice: %d chars for %s", len(result), filename)
        return result
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 429:
            logger.warning("transcribe_voice: Groq 429 rate-limit hit")
            raise RuntimeError("STT service is rate-limited. Try again in a moment.")
        if e.response.status_code == 401:
            logger.warning("transcribe_voice: Groq 401 auth error — check GROQ_API_KEY")
            raise RuntimeError("STT authentication error. Check GROQ_API_KEY.")
        logger.warning("transcribe_voice: Groq %d error body: %s", e.response.status_code, e.response.text)
        raise RuntimeError(f"STT service error ({e.response.status_code}).")
    except httpx.TimeoutException:
        raise RuntimeError("STT service timed out. Please try again.")