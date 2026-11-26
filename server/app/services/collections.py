"""Collections: provider registry — the extension point for future feeds.

Adding a new collection (e.g. a recommendation feed) = one entry in
REGISTRY with a page() implementation; GET /collections/{name}/tracks and
the registry listing pick it up automatically.
"""

from __future__ import annotations

from typing import Any

from .. import db
from .dto import build_track_dtos, page

# name -> {type, description}
# 'recommendations' is a RESERVED stub: GET /collections/{name}/tracks
# returns 501 until the recommendation engine lands (see AGENTS.md).
REGISTRY: dict[str, dict[str, str]] = {
    "favorites": {
        "type": "system",
        "description": "Tracks the user liked (download-on-like).",
    },
    "recommendations": {
        "type": "reserved",
        "description": "Reserved for the future recommendation engine.",
    },
}


async def list_collections() -> list[dict[str, str]]:
    return [{"name": name, **meta} for name, meta in REGISTRY.items()]


async def is_available(name: str) -> bool:
    return name in REGISTRY and name != "recommendations"


async def favorites_page(
    user_id: int, limit: int, offset: int, sort: str, query: str | None = None
) -> dict[str, Any]:
    order = {
        "added": "f.created_at DESC",
        "added_asc": "f.created_at ASC",
        "title": "lower(t.title)",
        "artist": "lower(array_to_string(t.artists, ', '))",
    }.get(sort, "f.created_at DESC")
    needle = f"%{(query or '').strip()}%"
    params: dict[str, Any] = {"u": user_id, "l": limit, "o": offset}
    where = "WHERE f.user_id = :u"
    if (query or "").strip():
        where += (
            " AND (t.title ILIKE :q"
            " OR t.artists_text ILIKE :q"
            " OR COALESCE(t.album, '') ILIKE :q)"
        )
        params["q"] = needle
    total = await db.fetch_one(
        f"""
        SELECT COUNT(*) AS n
        FROM favorites f JOIN tracks t ON t.id = f.track_id
        {where}
        """,
        params,
    )
    rows = await db.fetch_all(
        f"""
        SELECT f.track_id, t.title, t.artists, t.duration_sec FROM favorites f
        JOIN tracks t ON t.id = f.track_id
        {where}
        ORDER BY {order}
        LIMIT :l OFFSET :o
        """,
        params,
    )
    # Transition-period safety: a twin row of the same recording may still
    # carry its own old like. Collapse the page by identity so the library
    # never shows the same song twice (like() + repair --merge prevent new
    # twins from appearing at all).
    from .identity import identity_hash as _identity_hash

    seen: dict[str, list[int]] = {}
    ids: list[int] = []
    for r in rows:
        h = _identity_hash(r["title"], r["artists"])
        dur = r.get("duration_sec") or 0
        durs = seen.setdefault(h, [])
        if any(abs(dur - d) <= 15 for d in durs):
            continue
        durs.append(dur)
        ids.append(int(r["track_id"]))
    items = await build_track_dtos(ids, user_id)
    return page(items, total["n"] if total else 0, limit, offset)
