"""Secrets envelope + quality-tier sanity (offline, no DB)."""

from __future__ import annotations

import os

os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://kwhale:kwhale_v3_pg@localhost:5433/kwhale")

from app.services import secrets as sec  # noqa: E402


def test_encrypt_decrypt_roundtrip():
    blob = sec.encrypt_text('{"token": "abc123"}')
    assert blob.startswith("gcm1:")
    assert "abc123" not in blob
    assert sec.decrypt_text(blob) == '{"token": "abc123"}'


def test_envelope_is_nondeterministic():
    a = sec.encrypt_text("same")
    b = sec.encrypt_text("same")
    assert a != b  # fresh nonce every time


def test_tamper_detected():
    blob = sec.encrypt_text("payload")
    import base64

    raw = bytearray(base64.b64decode(blob[len("gcm1=") :]))
    raw[-1] ^= 0xFF
    tampered = "gcm1:" + base64.b64encode(bytes(raw)).decode()
    try:
        sec.decrypt_text(tampered)
    except Exception:
        return
    raise AssertionError("tampered envelope decrypted")
