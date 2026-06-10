"""JWT authentication + bcrypt password hashing for tenant portal.

Portal sessions use HS256 JWTs (symmetric — we control both sides).
Short TTL + per-request tenant.active re-check for revocation support.
"""
import time

import bcrypt
import jwt

JWT_ALGORITHM = "HS256"
JWT_TTL_SECONDS = 3600  # 1 hour


def create_access_token(slug: str, signing_secret: str) -> str:
    """Create a JWT for the given tenant slug.

    Args:
        slug: Tenant slug (becomes the JWT subject).
        signing_secret: HS256 signing key from settings.jwt_secret.

    Returns:
        Encoded JWT string.
    """
    now = time.time()
    payload = {
        "sub": slug,
        "iat": int(now),
        "exp": int(now) + JWT_TTL_SECONDS,
    }
    return jwt.encode(payload, signing_secret, algorithm=JWT_ALGORITHM)


def decode_access_token(token: str, signing_secret: str) -> dict:
    """Decode and verify a JWT.

    Raises jwt.InvalidTokenError (or subclass) on invalid/expired tokens.
    """
    return jwt.decode(token, signing_secret, algorithms=[JWT_ALGORITHM])


def verify_portal_password(plain: str, hashed: str) -> bool:
    """Verify a portal password against its bcrypt hash."""
    return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))


def hash_portal_password(plain: str) -> str:
    """Hash a portal password with bcrypt."""
    return bcrypt.hashpw(plain.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")