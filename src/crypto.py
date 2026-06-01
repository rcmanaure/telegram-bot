"""
Symmetric encryption for secrets stored in the database.

Uses Fernet (AES-128-CBC + HMAC-SHA256 from the `cryptography` package).
ENCRYPTION_KEY must be a URL-safe base64-encoded 32-byte key.

Generate a key:
    python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"

If ENCRYPTION_KEY is not set, values are stored/returned as plaintext with a warning.
If decryption fails (e.g., value was stored before encryption was enabled),
decrypt_value() returns the input unchanged (plaintext fallback).
"""

import logging
import os

logger = logging.getLogger(__name__)

_fernet = None
_warned_missing_key = False


def get_fernet():
    """Return a lazy-initialised Fernet instance. Returns None if key not configured."""
    global _fernet, _warned_missing_key
    if _fernet is not None:
        return _fernet
    key = os.environ.get("ENCRYPTION_KEY", "")
    if not key:
        if not _warned_missing_key:
            logger.warning(
                "ENCRYPTION_KEY not set — secrets stored/read as plaintext. "
                "Generate a key: python -c \"from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())\""
            )
            _warned_missing_key = True
        return None
    try:
        from cryptography.fernet import Fernet
        _fernet = Fernet(key.encode())
        return _fernet
    except Exception as e:
        logger.error("ENCRYPTION_KEY is invalid: %s", e)
        return None


def encrypt_value(plaintext: str) -> str:
    """Encrypt a string. Returns Fernet token (base64) or plaintext if key not set."""
    if not plaintext:
        return plaintext
    f = get_fernet()
    if f is None:
        return plaintext
    return f.encrypt(plaintext.encode()).decode()


def decrypt_value(ciphertext: str) -> str:
    """Decrypt a Fernet token. Falls back to returning input as-is if decryption fails.
    This handles values stored before encryption was enabled (backward compat)."""
    if not ciphertext:
        return ciphertext
    f = get_fernet()
    if f is None:
        return ciphertext
    try:
        return f.decrypt(ciphertext.encode()).decode()
    except Exception:
        # Value is likely plaintext from before encryption was enabled
        return ciphertext


def generate_key() -> str:
    """Generate a new Fernet key. Paste the output into ENCRYPTION_KEY in .env."""
    from cryptography.fernet import Fernet
    return Fernet.generate_key().decode()
