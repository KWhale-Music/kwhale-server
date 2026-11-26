"""Playlists: user CRUD, visibility (private/unlisted/public) and follows.

Access rules:
  * private  — owner only;
  * unlisted — accessible by direct link, not searchable, followable;
  * public   — searchable, followable.
Following someone else's playlist is read-only: tracks are managed by the owner.
"""

from __future__ import annotations

from typing import Any

from .. import db
from .dto import build_track_dtos, page

VISIBILITIES = ("private", "unlisted", "public")

_SELECT = """
SELECT p.*,
       (SELECT COUNT(*) FROM playlist_tracks pt WHERE pt.playlist_id = p.id) AS track_count,
       (SELECT COUNT(*) FROM playlist_follows pf WHERE pf.playlist_id = p.id) AS followers_count,
       EXISTS (
           SELECT 1 FROM playlist_follows pf2
           WHERE pf2.playlist_id = p.id AND pf2.user_id = :u
       ) AS is_following
FROM playlists p
"""


def playlist_dto(
    row: dict[str, Any],
    track_count: int | None = None,
    *,
    is_following: bool | None = None,
) -> dict[str, Any]:
    created = row.get("created_at")
    updated = row.get("updated_at")
    dto = {
        "id": row["id"],
        "user_id": row["user_id"],
        "name": row["name"],
        "description": row.get("description"),
        "is_system": row["is_system"],
        "source": row["source"],
        "created_at": created.isoformat() if created else None,
        "updated_at": updated.isoformat() if updated else None,
        "visibility": row.get("visibility") or "private",
        "cover_image_id": row.get("cover_image_id"),
        "followers_count": row.get("followers_count") or 0,
        "is_following": bool(
            row.get("is_following") if is_following is None else is_following
        ),
    }
    if track_count is not None:
        dto["track_count"] = track_count
    return dto


# ── access ───────────────────────────────────────────────────────────────────


async def _row(playlist_id: int, user_id: int) -> dict[str, Any] | None:
    """Playlist row if visible to the user (own, unlisted or public)."""
    return await db.fetch_one(
        _SELECT
        + """
        WHERE p.id = :i
          AND (p.user_id = :u OR p.visibility IN ('unlisted', 'public'))
        """,
        {"i": playlist_id, "u": user_id},
    )


async def exists(playlist_id: int) -> bool:
    row = await db.fetch_one("SELECT id FROM playlists WHERE id = :i", {"i": playlist_id})
    return row is not None


async def _owns(playlist_id: int, user_id: int) -> bool:
    row = await db.fetch_one(
        "SELECT id FROM playlists WHERE id = :i AND user_id = :u",
        {"i": playlist_id, "u": user_id},
    )
    return row is not None


# ── CRUD ─────────────────────────────────────────────────────────────────────


async def create(
    user_id: int,
    name: str,
    description: str | None = None,
    visibility: str = "private",
    *,
    is_system: bool = False,
    source: str = "user",
) -> dict[str, Any] | None:
    rows = await db.execute_returning(
        """
        INSERT INTO playlists (user_id, name, description, visibility, is_system, source)
        VALUES (:u, :n, :d, :v, :sys, :src)
        RETURNING id
        """,
        {
            "u": user_id,
            "n": name,
            "d": description,
            "v": visibility,
            "sys": is_system,
            "src": source,
        },
    )
    return await get_meta(rows[0]["id"], user_id)


async def get_meta(playlist_id: int, user_id: int) -> dict[str, Any] | None:
    row = await _row(playlist_id, user_id)
    return playlist_dto(row, row["track_count"]) if row else None


async def list_playlists(
    user_id: int, limit: int, offset: int, scope: str = "own"
) -> dict[str, Any]:
    if scope == "following":
        where = """
            JOIN playlist_follows f
              ON f.playlist_id = p.id AND f.user_id = :u
        """
        total_sql = """
            SELECT COUNT(*) AS n FROM playlists p
            JOIN playlist_follows f ON f.playlist_id = p.id AND f.user_id = :u
        """
    elif scope == "library":
        visible = """
            (p.user_id = :u
             OR EXISTS (
                 SELECT 1 FROM playlist_follows f
                 WHERE f.playlist_id = p.id AND f.user_id = :u
             ))
            AND NOT (
                p.source = 'recommendation' AND p.is_system = TRUE
                AND NOT EXISTS (SELECT 1 FROM playlist_tracks pt WHERE pt.playlist_id = p.id)
            )
        """
        where = f"WHERE {visible}"
        total_sql = f"SELECT COUNT(*) AS n FROM playlists p WHERE {visible}"
    else:
        where = "WHERE p.user_id = :u"
        total_sql = "SELECT COUNT(*) AS n FROM playlists p WHERE p.user_id = :u"

    total = await db.fetch_one(total_sql, {"u": user_id})
    rows = await db.fetch_all(
        _SELECT + where + " ORDER BY p.updated_at DESC LIMIT :l OFFSET :o",
        {"u": user_id, "l": limit, "o": offset},
    )
    items = [playlist_dto(r, r["track_count"]) for r in rows]
    return page(items, total["n"] if total else 0, limit, offset)


async def update(
    playlist_id: int,
    user_id: int,
    name: str | None,
    description: str | None,
    visibility: str | None = None,
) -> dict[str, Any] | None:
    if not await _owns(playlist_id, user_id):
        return None
    await db.execute(
        """
        UPDATE playlists SET
            name = COALESCE(:n, name),
            description = COALESCE(:d, description),
            visibility = COALESCE(:v, visibility),
            updated_at = NOW()
        WHERE id = :i
        """,
        {"n": name, "d": description, "v": visibility, "i": playlist_id},
    )
    return await get_meta(playlist_id, user_id)


async def delete(playlist_id: int, user_id: int) -> bool:
    rows = await db.execute_returning(
        "DELETE FROM playlists WHERE id = :i AND user_id = :u RETURNING id",
        {"i": playlist_id, "u": user_id},
    )
    return bool(rows)


# ── tracks ───────────────────────────────────────────────────────────────────


async def add_track(playlist_id: int, user_id: int, track_id: int) -> int | None:
    """Append a track. Returns next position or None (not found / not owner)."""
    if not await _owns(playlist_id, user_id):
        return None
    track = await db.fetch_one("SELECT id FROM tracks WHERE id = :t", {"t": track_id})
    if not track:
        return None
    pos_row = await db.fetch_one(
        "SELECT COALESCE(MAX(position), 0) + 1 AS next FROM playlist_tracks WHERE playlist_id = :p",
        {"p": playlist_id},
    )
    next_pos = pos_row["next"] if pos_row else 1
    await db.execute(
        """
        INSERT INTO playlist_tracks (playlist_id, track_id, position)
        VALUES (:p, :t, :pos)
        ON CONFLICT (playlist_id, track_id) DO NOTHING
        """,
        {"p": playlist_id, "t": track_id, "pos": next_pos},
    )
    await db.execute("UPDATE playlists SET updated_at = NOW() WHERE id = :i", {"i": playlist_id})
    return next_pos


async def remove_track(playlist_id: int, user_id: int, track_id: int) -> bool:
    if not await _owns(playlist_id, user_id):
        return False
    rows = await db.execute_returning(
        "DELETE FROM playlist_tracks WHERE playlist_id = :p AND track_id = :t RETURNING track_id",
        {"p": playlist_id, "t": track_id},
    )
    if rows:
        await _renumber(playlist_id)
        await db.execute(
            "UPDATE playlists SET updated_at = NOW() WHERE id = :i", {"i": playlist_id}
        )
    return bool(rows)


async def reorder(playlist_id: int, user_id: int, track_ids: list[int]) -> bool:
    """Set the full order. Tracks not mentioned keep relative order at the end."""
    if not await _owns(playlist_id, user_id):
        return False
    current = await db.fetch_all(
        "SELECT track_id FROM playlist_tracks WHERE playlist_id = :p ORDER BY position",
        {"p": playlist_id},
    )
    current_ids = [r["track_id"] for r in current]
    ordered = [tid for tid in track_ids if tid in current_ids]
    ordered += [tid for tid in current_ids if tid not in set(track_ids)]
    for pos, tid in enumerate(ordered, start=1):
        await db.execute(
            "UPDATE playlist_tracks SET position = :pos WHERE playlist_id = :p AND track_id = :t",
            {"p": playlist_id, "t": tid, "pos": pos},
        )
    await db.execute("UPDATE playlists SET updated_at = NOW() WHERE id = :i", {"i": playlist_id})
    return True


async def _renumber(playlist_id: int) -> None:
    rows = await db.fetch_all(
        "SELECT track_id FROM playlist_tracks WHERE playlist_id = :p ORDER BY position",
        {"p": playlist_id},
    )
    for pos, r in enumerate(rows, start=1):
        await db.execute(
            "UPDATE playlist_tracks SET position = :pos WHERE playlist_id = :p AND track_id = :t",
            {"pos": pos, "p": playlist_id, "t": r["track_id"]},
        )


async def list_visible(user_id: int, limit: int, offset: int) -> dict[str, Any]:
    """All playlists visible to the user: own, public and followed."""
    total = await db.fetch_one(
        """
        SELECT COUNT(*) AS n FROM playlists p
        WHERE p.user_id = :u OR p.visibility = 'public'
           OR EXISTS (SELECT 1 FROM playlist_follows f
                      WHERE f.playlist_id = p.id AND f.user_id = :u)
        """,
        {"u": user_id},
    )
    rows = await db.fetch_all(
        _SELECT
        + """
        WHERE p.user_id = :u OR p.visibility = 'public'
           OR EXISTS (SELECT 1 FROM playlist_follows f
                      WHERE f.playlist_id = p.id AND f.user_id = :u)
        ORDER BY p.updated_at DESC LIMIT :l OFFSET :o
        """,
        {"u": user_id, "l": limit, "o": offset},
    )
    items = [playlist_dto(r, r["track_count"]) for r in rows]
    return page(items, total["n"] if total else 0, limit, offset)


async def add_tracks(
    playlist_id: int,
    user_id: int,
    track_ids: list[int],
    position: int | None = None,
) -> dict[str, Any] | None:
    """Bulk add tracks; position is 0-based. Idempotent per track."""
    if not await _owns(playlist_id, user_id):
        return None
    valid: list[int] = []
    for tid in track_ids:
        row = await db.fetch_one("SELECT id FROM tracks WHERE id = :t", {"t": tid})
        if row:
            valid.append(tid)
    if not valid:
        return await get_meta(playlist_id, user_id)
    max_row = await db.fetch_one(
        "SELECT COALESCE(MAX(position), 0) AS n FROM playlist_tracks WHERE playlist_id = :p",
        {"p": playlist_id},
    )
    max_pos = int((max_row or {}).get("n") or 0)
    if position is None or position < 0 or position > max_pos:
        start = max_pos + 1
    else:
        start = position + 1
        await db.execute(
            "UPDATE playlist_tracks SET position = position + :n "
            "WHERE playlist_id = :p AND position >= :pos",
            {"n": len(valid), "p": playlist_id, "pos": start},
        )
    for idx, tid in enumerate(valid):
        await db.execute(
            """
            INSERT INTO playlist_tracks (playlist_id, track_id, position)
            VALUES (:p, :t, :pos)
            ON CONFLICT (playlist_id, track_id) DO NOTHING
            """,
            {"p": playlist_id, "t": tid, "pos": start + idx},
        )
    await _renumber(playlist_id)
    await db.execute("UPDATE playlists SET updated_at = NOW() WHERE id = :i", {"i": playlist_id})
    return await get_meta(playlist_id, user_id)


async def list_tracks(
    playlist_id: int, user_id: int, limit: int, offset: int
) -> dict[str, Any] | None:
    # public/unlisted playlists are readable by others; private by owner only
    if not await _row(playlist_id, user_id):
        return None
    total = await db.fetch_one(
        "SELECT COUNT(*) AS n FROM playlist_tracks WHERE playlist_id = :p",
        {"p": playlist_id},
    )
    rows = await db.fetch_all(
        """
        SELECT track_id FROM playlist_tracks
        WHERE playlist_id = :p
        ORDER BY position
        LIMIT :l OFFSET :o
        """,
        {"p": playlist_id, "l": limit, "o": offset},
    )
    ids = [r["track_id"] for r in rows]
    items = await build_track_dtos(ids, user_id)
    return page(items, total["n"] if total else 0, limit, offset)


# ── follows ──────────────────────────────────────────────────────────────────


async def follow(playlist_id: int, user_id: int) -> dict[str, Any]:
    row = await _row(playlist_id, user_id)
    if not row:
        return {"ok": False, "error": "not_found"}
    if row["user_id"] == user_id:
        return {"ok": False, "error": "own_playlist"}
    await db.execute(
        """
        INSERT INTO playlist_follows (user_id, playlist_id)
        VALUES (:u, :p)
        ON CONFLICT DO NOTHING
        """,
        {"u": user_id, "p": playlist_id},
    )
    return {"ok": True, "playlist_id": playlist_id, "following": True}


async def unfollow(playlist_id: int, user_id: int) -> dict[str, Any]:
    await db.execute(
        "DELETE FROM playlist_follows WHERE user_id = :u AND playlist_id = :p",
        {"u": user_id, "p": playlist_id},
    )
    return {"playlist_id": playlist_id, "following": False}


async def search_playlists(query: str, user_id: int, limit: int) -> list[dict[str, Any]]:
    rows = await db.fetch_all(
        _SELECT
        + """
        WHERE (p.user_id = :u OR p.visibility = 'public')
          AND p.name ILIKE :q
        ORDER BY p.updated_at DESC
        LIMIT :l
        """,
        {"u": user_id, "q": f"%{query}%", "l": limit},
    )
    return [playlist_dto(r, r["track_count"]) for r in rows]
