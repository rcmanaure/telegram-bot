"""Tests for JWT authentication and bcrypt password hashing (src/auth.py)."""
import time

import jwt as pyjwt
import pytest

from auth import (
    JWT_ALGORITHM,
    create_access_token,
    decode_access_token,
    hash_portal_password,
    verify_portal_password,
)


# ─── JWT ─────────────────────────────────────────────────────────────────────

class TestJWT:
    SECRET = "test-secret-key-for-jwt-tests"

    def test_create_and_decode_roundtrip(self):
        token = create_access_token("acme-pharmacy", self.SECRET)
        payload = decode_access_token(token, self.SECRET)
        assert payload["sub"] == "acme-pharmacy"
        assert "iat" in payload
        assert "exp" in payload

    def test_expired_token_rejected(self):
        # Manually craft an expired token
        payload = {
            "sub": "acme-pharmacy",
            "iat": int(time.time()) - 3600,
            "exp": int(time.time()) - 1,
        }
        token = pyjwt.encode(payload, self.SECRET, algorithm=JWT_ALGORITHM)
        with pytest.raises(pyjwt.ExpiredSignatureError):
            decode_access_token(token, self.SECRET)

    def test_wrong_secret_rejected(self):
        token = create_access_token("acme-pharmacy", self.SECRET)
        with pytest.raises(pyjwt.InvalidSignatureError):
            decode_access_token(token, "wrong-secret")

    def test_tampered_payload_rejected(self):
        token = create_access_token("acme-pharmacy", self.SECRET)
        # Tamper with the token by flipping a character
        tampered = token[:-5] + ("x" if token[-5] != "x" else "y") + token[-4:]
        with pytest.raises(pyjwt.InvalidTokenError):
            decode_access_token(tampered, self.SECRET)

    def test_different_slugs_different_tokens(self):
        t1 = create_access_token("tenant-a", self.SECRET)
        t2 = create_access_token("tenant-b", self.SECRET)
        assert t1 != t2
        assert decode_access_token(t1, self.SECRET)["sub"] == "tenant-a"
        assert decode_access_token(t2, self.SECRET)["sub"] == "tenant-b"


# ─── Bcrypt ───────────────────────────────────────────────────────────────────

class TestBcrypt:
    def test_hash_and_verify_roundtrip(self):
        hashed = hash_portal_password("supersecret123")
        assert verify_portal_password("supersecret123", hashed)

    def test_wrong_password_fails(self):
        hashed = hash_portal_password("correct-password")
        assert not verify_portal_password("wrong-password", hashed)

    def test_hash_is_different_each_time(self):
        h1 = hash_portal_password("same-password")
        h2 = hash_portal_password("same-password")
        assert h1 != h2  # bcrypt generates unique salts
        # Both should verify correctly
        assert verify_portal_password("same-password", h1)
        assert verify_portal_password("same-password", h2)

    def test_empty_password(self):
        hashed = hash_portal_password("")
        assert verify_portal_password("", hashed)
        assert not verify_portal_password("notempty", hashed)