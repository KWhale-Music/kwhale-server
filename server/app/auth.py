"""Authentication: email/password, short access JWT, rotating refresh tokens,
invite-only registration and admin roles."""

from __future__ import annotations

import base64
import hashlib
import hmac
import os
import secrets
import time
from collections import OrderedDict, defaultdict
from datetime import UTC, datetime, timedelta
from typing import Any

import jwt
from fastapi import Depends, Query, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

from . import db
from .config import settings
from .errors import ProblemError

bearer = HTTPBearer(auto_error=False)


# scrypt (stdlib, memory-hard) replaces the old single-round salted SHA-256.
# Stored format: scrypt$<n>$<r>$<p>$<salt_b64>$<hash_b64>. Legacy rows use
# <salt>$<sha256hex> and are transparently upgraded on the next successful login.
_SCRYPT_N = 2**14
_SCRYPT_R = 8
_SCRYPT_P = 1
_SCRYPT_LEN = 32
_PREFIX = "scrypt$"


def _scrypt_hash(password: str, salt: bytes, n: int, r: int, p: int) -> bytes:
    return hashlib.scrypt(
        password.encode(), salt=salt, n=n, r=r, p=p, dklen=_SCRYPT_LEN, maxmem=64 * 1024 * 1024
    )


def hash_password(password: str) -> str:
    salt = os.urandom(16)
    digest = _scrypt_hash(password, salt, _SCRYPT_N, _SCRYPT_R, _SCRYPT_P)
    return (
        f"{_PREFIX}{_SCRYPT_N}${_SCRYPT_R}${_SCRYPT_P}$"
        f"{base64.b64encode(salt).decode()}${base64.b64encode(digest).decode()}"
    )


def _verify_scrypt(password: str, stored: str) -> bool:
    try:
        n, r, p, salt_b64, hash_b64 = stored[len(_PREFIX) :].split("$")
        computed = _scrypt_hash(
            password, base64.b64decode(salt_b64), int(n), int(r), int(p)
        )
        expected = base64.b64decode(hash_b64)
    except (ValueError, TypeError):
        return False
    return hmac.compare_digest(computed, expected)


def _verify_legacy(password: str, stored: str) -> bool:
    try:
        salt, digest = stored.split("$", 1)
    except ValueError:
        return False
    recomputed = hashlib.sha256(f"{salt}:{password}".encode()).hexdigest()
    return hmac.compare_digest(recomputed, digest)


def verify_password(password: str, stored: str) -> bool:
    if stored.startswith(_PREFIX):
        return _verify_scrypt(password, stored)
    return _verify_legacy(password, stored)


def password_needs_rehash(stored: str) -> bool:
    """True when the stored hash is still the weak legacy format."""
    return not stored.startswith(_PREFIX)


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def create_token(user_id: int, username: str, role: str = "user") -> str:
    """Backward-compatible access token (old tests import this)."""
    expire = datetime.now(UTC) + timedelta(minutes=settings.jwt_access_minutes)
    return jwt.encode(
        {"sub": str(user_id), "username": username, "role": role, "exp": expire},
        settings.jwt_secret,
        algorithm="HS256",
    )


def create_access_token(user_id: int, email: str, role: str) -> str:
    expire = datetime.now(UTC) + timedelta(minutes=settings.jwt_access_minutes)
    return jwt.encode(
        {"sub": str(user_id), "email": email, "role": role, "exp": expire},
        settings.jwt_secret,
        algorithm="HS256",
    )


def public_user(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": str(row["id"]),
        "email": row.get("email"),
        "displayName": row.get("display_name") or row.get("email") or row.get("username"),
        "avatar": {"id": "placeholder"},
        "createdAt": row["created_at"].isoformat() if row.get("created_at") else None,
    }


async def _issue_tokens(row: dict[str, Any], device_id: str | None = None) -> dict[str, Any]:
    user_id = row["id"]
    access = create_access_token(user_id, row.get("email") or "", row.get("role") or "user")
    refresh = secrets.token_urlsafe(48)
    await db.execute(
        """
        INSERT INTO refresh_tokens (user_id, token_hash, device_id, expires_at)
        VALUES (:u, :h, :d, NOW() + make_interval(days => :days))
        """,
        {
            "u": user_id,
            "h": hash_token(refresh),
            "d": device_id,
            "days": settings.refresh_token_days,
        },
    )
    return {
        "accessToken": access,
        "refreshToken": refresh,
        "expiresIn": settings.jwt_access_minutes * 60,
        "tokenType": "Bearer",
        "user": public_user(row),
    }


class LoginRequest(BaseModel):
    email: str = Field(min_length=3, max_length=320)
    password: str = Field(min_length=1, max_length=128)
    deviceId: str | None = None


class RegisterRequest(BaseModel):
    email: str = Field(min_length=3, max_length=320)
    password: str = Field(min_length=8, max_length=128)
    displayName: str = Field(min_length=1, max_length=80)
    inviteToken: str = Field(min_length=8)
    deviceId: str | None = None


class RefreshRequest(BaseModel):
    refreshToken: str = Field(min_length=8)


class LogoutRequest(BaseModel):
    refreshToken: str = Field(min_length=8)


async def ensure_seed_user() -> None:
    """Bootstrap the first admin if missing. Create-only: never resets an existing
    account's password/role (so an admin can change their password, and an invited
    user who happens to use the seed email is not silently promoted)."""
    email = settings.seed_email.strip().lower()
    row = await db.fetch_one(
        "SELECT 1 FROM users WHERE lower(email) = lower(:e) OR username = :u LIMIT 1",
        {"e": email, "u": settings.seed_username},
    )
    if row is not None:
        return
    if not settings.seed_password:
        raise RuntimeError("SEED_PASSWORD is empty — refusing to create an admin with no password")
    await db.execute(
        """
        INSERT INTO users (username, password_hash, email, display_name, role)
        VALUES (:u, :p, :e, :d, 'admin')
        """,
        {
            "u": settings.seed_username,
            "p": hash_password(settings.seed_password),
            "e": email,
            "d": settings.seed_display_name,
        },
    )


# ── login brute-force throttle (per IP+account) ─────────────────────────────
_login_failures: dict[str, list[float]] = defaultdict(list)
_LOGIN_MAX_FAILURES = 5
_LOGIN_WINDOW_SEC = 300.0


def check_login_throttle(key: str) -> None:
    now = time.monotonic()
    recent = [t for t in _login_failures[key] if now - t < _LOGIN_WINDOW_SEC]
    _login_failures[key] = recent
    if len(recent) >= _LOGIN_MAX_FAILURES:
        raise ProblemError(429, "rate_limited", "Слишком много попыток входа. Попробуйте позже.")


def record_login_failure(key: str) -> None:
    _login_failures[key].append(time.monotonic())


def clear_login_failures(key: str) -> None:
    _login_failures.pop(key, None)


async def login(body: LoginRequest) -> dict[str, Any]:
    row = await db.fetch_one(
        "SELECT * FROM users WHERE lower(email) = lower(:e) LIMIT 1",
        {"e": body.email.strip()},
    )
    if row is None or not verify_password(body.password, row["password_hash"]):
        raise ProblemError(401, "invalid_credentials", "Invalid email or password")
    if password_needs_rehash(row["password_hash"]):
        # transparent upgrade from legacy salted SHA-256 to scrypt on login
        await db.execute(
            "UPDATE users SET password_hash = :p WHERE id = :i",
            {"p": hash_password(body.password), "i": row["id"]},
        )
    return await _issue_tokens(row, body.deviceId)


async def register(body: RegisterRequest) -> dict[str, Any]:
    token_hash = hash_token(body.inviteToken.strip())
    invite = await db.fetch_one(
        """
        SELECT id FROM invites
        WHERE token_hash = :h AND used_at IS NULL AND revoked_at IS NULL
          AND expires_at > NOW()
        """,
        {"h": token_hash},
    )
    if invite is None:
        raise ProblemError(403, "invalid_invite", "Invite token is invalid or expired")

    email = body.email.strip().lower()
    if await db.fetch_one("SELECT id FROM users WHERE lower(email) = lower(:e)", {"e": email}):
        raise ProblemError(409, "conflict", "Email is already registered")

    base_username = (email.split("@")[0][:48] or "user").strip() or "user"
    username = base_username
    suffix = 0
    while await db.fetch_one("SELECT id FROM users WHERE username = :u", {"u": username}):
        suffix += 1
        username = f"{base_username}{suffix}"
    rows = await db.execute_returning(
        """
        INSERT INTO users (username, password_hash, email, display_name, role)
        VALUES (:u, :p, :e, :d, 'user')
        RETURNING *
        """,
        {
            "u": username,
            "p": hash_password(body.password),
            "e": email,
            "d": body.displayName.strip(),
        },
    )
    user = rows[0]
    await db.execute(
        "UPDATE invites SET used_by = :u, used_at = NOW() WHERE id = :i",
        {"u": user["id"], "i": invite["id"]},
    )
    return await _issue_tokens(user, body.deviceId)


async def refresh_session(body: RefreshRequest) -> dict[str, Any]:
    row = await db.fetch_one(
        """
        SELECT rt.id AS token_id, u.*
        FROM refresh_tokens rt
        JOIN users u ON u.id = rt.user_id
        WHERE rt.token_hash = :h AND rt.revoked_at IS NULL AND rt.expires_at > NOW()
        """,
        {"h": hash_token(body.refreshToken)},
    )
    if row is None:
        raise ProblemError(401, "token_expired", "Refresh token is invalid or expired")
    await db.execute(
        "UPDATE refresh_tokens SET revoked_at = NOW() WHERE id = :i",
        {"i": row["token_id"]},
    )
    return await _issue_tokens(row)


async def logout_session(body: LogoutRequest) -> None:
    await db.execute(
        "UPDATE refresh_tokens SET revoked_at = NOW() WHERE token_hash = :h",
        {"h": hash_token(body.refreshToken)},
    )


_USER_CACHE_TTL = 30.0
_USER_CACHE_MAX = 512
_user_cache: OrderedDict[int, tuple[float, dict[str, Any]]] = OrderedDict()


def clear_user_cache(user_id: int | None = None) -> None:
    """Clear cached user rows (single user or entire TTL cache)."""
    if user_id is None:
        _user_cache.clear()
    else:
        _user_cache.pop(user_id, None)


def sign_media_path(path: str, ttl_sec: int = 3600) -> tuple[int, str]:
    """Create a short-lived HMAC-SHA256 URL signature for cacheable media paths."""
    exp = int(time.time()) + max(1, int(ttl_sec))
    msg = f"{path}:{exp}".encode()
    key = b"kwhale-media-v1:" + settings.jwt_secret.encode()
    sig = hmac.new(key, msg, hashlib.sha256).hexdigest()[:32]
    return exp, sig


def verify_media_signature(path: str, exp: int | None, sig: str | None) -> bool:
    """Verify path-scoped HMAC signature and expiry timestamp."""
    if exp is None or not sig or exp < int(time.time()):
        return False
    msg = f"{path}:{exp}".encode()
    key = b"kwhale-media-v1:" + settings.jwt_secret.encode()
    expected = hmac.new(key, msg, hashlib.sha256).hexdigest()[:32]
    return hmac.compare_digest(expected, sig)


async def current_user(creds: HTTPAuthorizationCredentials = Depends(bearer)) -> dict[str, Any]:
    if creds is None:
        raise ProblemError(401, "invalid_credentials", "Not authenticated")
    try:
        payload = jwt.decode(creds.credentials, settings.jwt_secret, algorithms=["HS256"])
    except jwt.PyJWTError as e:
        raise ProblemError(401, "token_expired", "Access token is invalid or expired") from e
    user_id = int(payload["sub"])
    now = time.monotonic()
    cached = _user_cache.get(user_id)
    if cached is not None and cached[0] > now:
        _user_cache.move_to_end(user_id)
        return cached[1]
    row = await db.fetch_one("SELECT * FROM users WHERE id = :i", {"i": user_id})
    if row is None:
        _user_cache.pop(user_id, None)
        raise ProblemError(401, "invalid_credentials", "User no longer exists")
    row_dict = dict(row)
    _user_cache[user_id] = (now + _USER_CACHE_TTL, row_dict)
    _user_cache.move_to_end(user_id)
    while len(_user_cache) > _USER_CACHE_MAX:
        _user_cache.popitem(last=False)
    return row_dict


async def media_access(
    request: Request,
    creds: HTTPAuthorizationCredentials = Depends(bearer),
    exp: int | None = Query(None),
    sig: str | None = Query(None),
) -> dict[str, Any]:
    """Authenticate media streams via either Bearer JWT or signed (?exp=&sig=) query."""
    if exp is not None or sig is not None:
        if not verify_media_signature(request.url.path, exp, sig):
            raise ProblemError(401, "token_expired", "Media signature is invalid or expired")
        request.state.media_signed = True
        request.state.media_exp = exp
        return {"id": 0, "role": "signed"}
    request.state.media_signed = False
    return await current_user(creds)


async def require_admin(user: dict[str, Any] = Depends(current_user)) -> dict[str, Any]:
    if (user.get("role") or "user") != "admin":
        raise ProblemError(403, "forbidden", "Admin role required")
    return user


async def create_invite(created_by: int, days: int = 7) -> dict[str, Any]:
    raw = secrets.token_urlsafe(32)
    rows = await db.execute_returning(
        """
        INSERT INTO invites (token_hash, created_by, expires_at)
        VALUES (:h, :u, NOW() + make_interval(days => :days))
        RETURNING id, created_at, expires_at
        """,
        {"h": hash_token(raw), "u": created_by, "days": days},
    )
    inv = rows[0]
    return {
        "id": str(inv["id"]),
        "token": raw,
        "createdAt": inv["created_at"].isoformat(),
        "expiresAt": inv["expires_at"].isoformat(),
    }


async def list_invites() -> list[dict[str, Any]]:
    rows = await db.fetch_all(
        """
        SELECT i.id, i.created_at, i.expires_at, i.used_at, i.revoked_at,
               cu.display_name AS created_by_name, uu.display_name AS used_by_name
        FROM invites i
        LEFT JOIN users cu ON cu.id = i.created_by
        LEFT JOIN users uu ON uu.id = i.used_by
        ORDER BY i.id DESC
        """
    )
    return [
        {
            "id": str(r["id"]),
            "createdAt": r["created_at"].isoformat(),
            "expiresAt": r["expires_at"].isoformat(),
            "usedAt": r["used_at"].isoformat() if r["used_at"] else None,
            "revokedAt": r["revoked_at"].isoformat() if r["revoked_at"] else None,
            "createdBy": r["created_by_name"],
            "usedBy": r["used_by_name"],
            "status": "revoked"
            if r["revoked_at"]
            else ("used" if r["used_at"] else ("expired" if r["expires_at"] < datetime.now(UTC) else "active")),
        }
        for r in rows
    ]


async def revoke_invite(invite_id: int) -> bool:
    rows = await db.execute_returning(
        "UPDATE invites SET revoked_at = NOW() WHERE id = :i AND used_at IS NULL RETURNING id",
        {"i": invite_id},
    )
    return bool(rows)


async def list_users() -> list[dict[str, Any]]:
    rows = await db.fetch_all(
        "SELECT * FROM users ORDER BY id"
    )
    return [
        {
            "id": str(r["id"]),
            "email": r.get("email"),
            "displayName": r.get("display_name") or r.get("username"),
            "role": r.get("role") or "user",
            "createdAt": r["created_at"].isoformat() if r.get("created_at") else None,
        }
        for r in rows
    ]


async def set_role(user_id: int, role: str) -> bool:
    if role not in ("user", "admin"):
        raise ProblemError(400, "validation_failed", "role must be user or admin")
    row = await db.fetch_one("SELECT id FROM users WHERE id = :i", {"i": user_id})
    if row is None:
        return False
    if role == "user":
        admins = await db.fetch_one("SELECT COUNT(*) AS n FROM users WHERE role = 'admin'")
        current = await db.fetch_one("SELECT role FROM users WHERE id = :i", {"i": user_id})
        if current and current["role"] == "admin" and admins and admins["n"] <= 1:
            raise ProblemError(409, "conflict", "Cannot demote the last admin")
    await db.execute("UPDATE users SET role = :r WHERE id = :i", {"r": role, "i": user_id})
    return True
