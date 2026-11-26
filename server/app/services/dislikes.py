"""Dislikes: per-user negative signal.

Unlike favorites, a dislike has no download or library side effects — it only
marks the track for the user (a future recommendation layer will consume it).
"""

from __future__ import annotations

from typing import Any

from .. import db
from .dto import build_track_dtos, page


async def set_disliked(user_id: int, track_id: int, disliked: bool) -> bool:
    """Returns False when the track does not exist."""
    track = await db.fetch_one("SELECT id FROM tracks WHERE id = :t", {"t": track_id})
    if not track:
        return False
    if disliked:
        await db.execute(
            """
            INSERT INTO dislikes (user_id, track_id)
            VALUES (:u, :t)
            ON CONFLICT DO NOTHING
            """,
            {"u": user_id, "t": track_id},
        )
    else:
        await db.execute(
            "DELETE FROM dislikes WHERE user_id = :u AND track_id = :t",
            {"u": user_id, "t": track_id},
        )
    return True


async def list_dislikes(user_id: int, limit: int, offset: int) -> dict[str, Any]:
    total = await db.fetch_one(
        "SELECT COUNT(*) AS n FROM dislikes WHERE user_id = :u", {"u": user_id}
    )
    rows = await db.fetch_all(
        """
        SELECT track_id FROM dislikes
        WHERE user_id = :u
        ORDER BY created_at DESC
        LIMIT :l OFFSET :o
        """,
        {"u": user_id, "l": limit, "o": offset},
    )
    ids = [r["track_id"] for r in rows]
    items = await build_track_dtos(ids, user_id)
    return page(items, total["n"] if total else 0, limit, offset)
