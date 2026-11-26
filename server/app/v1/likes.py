"""Bulk-like helpers shared by album and playlist routes."""

from __future__ import annotations

from .. import db
from ..services import favorites as favorites_service
from .schemas import LikeAllResult


async def like_all(track_ids: list[int], user_id: int) -> LikeAllResult:
    requested = len(track_ids)
    liked = already = queued = failed = 0
    for track_id in track_ids:
        try:
            fav = await db.fetch_one(
                "SELECT 1 FROM favorites WHERE user_id = :u AND track_id = :t",
                {"u": user_id, "t": track_id},
            )
            if fav:
                already += 1
                continue
            local = await db.fetch_one(
                "SELECT 1 FROM track_sources WHERE track_id = :t AND connector = 'local'",
                {"t": track_id},
            )
            result = await favorites_service.like(user_id, track_id)
            if not result.get("ok"):
                failed += 1
                continue
            liked += 1
            if not local:
                queued += 1
        except Exception:
            failed += 1
    return LikeAllResult(
        requested=requested,
        liked=liked,
        already_liked=already,
        queued=queued,
        failed=failed,
    )
