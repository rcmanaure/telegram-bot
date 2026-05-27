import hashlib
import logging
import re
import secrets
import unicodedata

logger = logging.getLogger(__name__)

# Rotates at server startup — unpredictable to attackers.
# NOTE: single-worker only — each uvicorn worker gets a different token.
# Multi-worker deployments need CANARY_TOKEN centralized in Redis or DB.
CANARY_TOKEN = secrets.token_hex(8)

_INJECTION_PATTERNS = re.compile(
    r'(ignore\s+(all\s+)?(previous|prior|above)\s+(instructions?|prompts?|rules?)'
    r'|system\s*prompt'
    r'|you\s+are\s+now\s+\w+'
    r'|act\s+as\s+(if\s+you\s+are|a\s)'
    r'|jailbreak'
    r'|do\s+anything\s+now'
    r'|roleplay\s+as'
    r'|pretend\s+(you\s+are|to\s+be)'
    r'|forget\s+(everything|all)\s+(you|your)'
    r'|override\s+(your\s+)?(instructions?|rules?)'
    r'|from\s+now\s+on\s+(you|ignore)'
    r'|reveal\s+(your\s+)?(system\s+)?prompt'
    r'|print\s+(your\s+)?(system\s+)?prompt'
    r'|CANARY_KEY:)',
    re.IGNORECASE | re.DOTALL,
)

MAX_INPUT_CHARS = 2000


def sanitize_user_input(text: str) -> str:
    """Normalize and check user input for injection patterns.

    Returns the (possibly truncated) text.
    Raises ValueError if injection is detected.

    Detection runs on the FULL normalized text before truncation so that
    injections hidden past char 2000 are still logged even though the LLM
    never sees them.
    """
    if not text:
        return text

    text = unicodedata.normalize("NFC", text)

    if _INJECTION_PATTERNS.search(text):
        logger.warning(
            "prompt_injection_attempt input_hash=%s snippet=%r",
            hashlib.sha256(text.encode()).hexdigest()[:8],
            text[:50],
        )
        raise ValueError("Mensaje no permitido.")

    if len(text) > MAX_INPUT_CHARS:
        text = text[:MAX_INPUT_CHARS]

    return text


def scan_chunk_for_injection(content: str) -> bool:
    """Returns True if the chunk looks like an injection attempt.

    Call inside index_chunks() — skip chunks that return True.
    """
    return bool(_INJECTION_PATTERNS.search(content))


def validate_output(response: str, user_id: str = "") -> str:
    """Log a warning if the LLM response contains the canary token.

    Returns response unchanged — defense is detection, not blocking.
    """
    if CANARY_TOKEN in response:
        logger.warning(
            "canary_exfiltration_detected uid=%s response_snippet=%r",
            user_id,
            response[:100],
        )
    return response
