"""Secrets at rest: AES-256-GCM envelopes for source account payloads.

Payload (tokens, cookies) never leaves this module unencrypted: rows in
source_accounts hold `gcm1:<b64 nonce|ct|tag>`. Key comes from
ACCOUNTS_ENC_KEY; when unset we derive one from JWT_SECRET (dev fallback,
logged once) so the stack still boots without extra env.
"""

from __future__ import annotations

import base64
import hashlib
import logging
import os

log = logging.getLogger("kwhale.secrets")

_PREFIX = "gcm1:"
_NONCE_LEN = 12
_TAG_LEN = 16

_warned_insecure = False


def _key() -> bytes:
    global _warned_insecure
    from ..config import settings

    raw = settings.accounts_enc_key
    if not raw:
        if not _warned_insecure:
            _warned_insecure = True
            log.warning(
                "ACCOUNTS_ENC_KEY not set — deriving from JWT_SECRET (dev only, set a real key)"
            )
        raw = settings.jwt_secret
    return hashlib.sha256(raw.encode()).digest()


def encrypt_text(plain: str) -> str:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    nonce = os.urandom(_NONCE_LEN)
    ct = AESGCM(_key()).encrypt(nonce, plain.encode("utf-8"), None)
    return _PREFIX + base64.b64encode(nonce + ct).decode("ascii")


def decrypt_text(blob: str) -> str:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    if not blob.startswith(_PREFIX):
        raise ValueError("not a kwhale secret envelope")
    raw = base64.b64decode(blob[len(_PREFIX) :])
    if len(raw) < _NONCE_LEN + _TAG_LEN:
        raise ValueError("secret envelope truncated")
    nonce, ct = raw[:_NONCE_LEN], raw[_NONCE_LEN:]
    plain = AESGCM(_key()).decrypt(nonce, ct, None)
    return plain.decode("utf-8")
