"""Shared KWhale v1 request helpers."""

from __future__ import annotations

import base64
import json
from typing import Any

from .. import db
from ..errors import ProblemError


def parse_id(value: str, field: str = "id") -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        raise ProblemError(404, "not_found", f"{field} not found") from None


def page_params(cursor: str | None, limit: int | None) -> tuple[int, int]:
    if limit is None:
        limit = 50
    if limit < 1 or limit > 100:
        raise ProblemError(400, "validation_failed", "limit must be between 1 and 100")
    offset = 0
    if cursor:
        try:
            raw = base64.urlsafe_b64decode(cursor.encode()).decode()
            if not raw.startswith("o:"):
                raise ValueError
            offset = int(raw[2:])
        except Exception:
            raise ProblemError(400, "validation_failed", "invalid cursor") from None
    return limit, offset


def make_cursor(offset: int, total: int, limit: int) -> str | None:
    if offset + limit >= total:
        return None
    return base64.urlsafe_b64encode(f"o:{offset + limit}".encode()).decode()


async def idempotent_get(user_id: int, key: str | None, endpoint: str) -> Any | None:
    if not key:
        return None
    row = await db.fetch_one(
        "SELECT response FROM idempotency_keys WHERE key = :k AND user_id = :u AND endpoint = :e",
        {"k": key, "u": user_id, "e": endpoint},
    )
    if row is None:
        return None
    try:
        return row["response"]
    except Exception:
        return None


async def idempotent_put(user_id: int, key: str | None, endpoint: str, response: Any) -> None:
    if not key:
        return
    await db.execute(
        """
        INSERT INTO idempotency_keys (key, user_id, endpoint, response)
        VALUES (:k, :u, :e, :r)
        ON CONFLICT (user_id, endpoint, key) DO NOTHING
        """,
        {"k": key, "u": user_id, "e": endpoint, "r": json.dumps(response)},
    )
