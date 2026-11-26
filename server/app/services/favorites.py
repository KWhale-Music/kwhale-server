"""Favorites: personal likes + shared local library.

Like:    insert favorite + enqueue download (no re-download if file exists).
Unlike:  remove favorite; if nobody likes the track anymore AND it has a
         local file -> set pending_delete_since (GC removes it after grace).

Albums and artists can be liked too: that only saves the card to the library
(no download), see like_album / like_artist.
"""

from __future__ import annotations

import logging
from typing import Any

from .. import db
from ..config import settings
from . import downloader
from .entity_upsert import ensure_entities_for_track

log = logging.getLogger("kwhale.favorites")


async def _cluster_ids(track_id: int) -> list[int]:
    """All rows of the same recording (identity + ±15 s), including itself."""
    row = await db.fetch_one(
        "SELECT id, identity_hash, duration_sec FROM tracks WHERE id = :t",
        {"t": track_id},
    )
    if not row:
        return [track_id]
    if not row.get("identity_hash"):
        return [track_id]
    rows = await db.fetch_all(
        """
        SELECT id, duration_sec FROM tracks
        WHERE identity_hash = :ih AND identity_hash IS NOT NULL
        """,
        {"ih": row["identity_hash"]},
    )
    from .identity import duration_close

    return [
        int(r["id"])
        for r in rows
        if duration_close(row["duration_sec"], r.get("duration_sec"))
    ] or [track_id]


async def like(user_id: int, track_id: int) -> dict[str, Any]:
    track = await db.fetch_one("SELECT id FROM tracks WHERE id = :t", {"t": track_id})
    if not track:
        return {"ok": False, "error": "track not found"}
    await db.execute(
        """
        INSERT INTO favorites (user_id, track_id)
        VALUES (:u, :t)
        ON CONFLICT DO NOTHING
        """,
        {"u": user_id, "t": track_id},
    )
    # the recording may have twin rows that carry an older like of this user —
    # keep exactly one favorite row per cluster so the library never shows
    # duplicates («в библиотеке появляются дубли»).
    cluster = await _cluster_ids(track_id)
    placeholders = [f":c{i}" for i in range(len(cluster))]
    params: dict[str, Any] = {"u": user_id, "t": track_id}
    params.update({f"c{i}": cid for i, cid in enumerate(cluster)})
    await db.execute(
        f"""
        DELETE FROM favorites WHERE user_id = :u AND track_id != :t
          AND track_id IN ({", ".join(placeholders)})
        """,
        params,
    )
    # someone liked it again -> cancel pending file deletion across the cluster
    await db.execute(
        f"""
        UPDATE track_sources SET pending_delete_since = NULL
        WHERE connector = 'local' AND track_id IN ({", ".join(placeholders)})
        """,
        params,
    )
    # materialise artist/album rows so the track can navigate to its cards
    try:
        await ensure_entities_for_track(track_id)
    except Exception:
        log.warning("entity ensure failed for track %s", track_id, exc_info=True)
    await downloader.enqueue_if_needed(track_id)
    return {"ok": True}


async def unlike(user_id: int, track_id: int) -> dict[str, Any]:
    # The like follows the recording: removing it must clear every twin row of
    # the cluster, otherwise the track keeps showing as liked from a sibling.
    cluster = await _cluster_ids(track_id)
    placeholders = [f":c{i}" for i in range(len(cluster))]
    params: dict[str, Any] = {"u": user_id}
    params.update({f"c{i}": cid for i, cid in enumerate(cluster)})
    await db.execute(
        f"DELETE FROM favorites WHERE user_id = :u AND track_id IN ({', '.join(placeholders)})",
        params,
    )
    remaining = await db.fetch_one(
        f"SELECT COUNT(*) AS n FROM favorites WHERE track_id IN ({', '.join(placeholders)})",
        params,
    )
    if remaining and remaining["n"] == 0:
        await db.execute(
            f"""
            UPDATE track_sources SET pending_delete_since = NOW()
            WHERE connector = 'local' AND permanent = FALSE
              AND pending_delete_since IS NULL
              AND track_id IN ({", ".join(placeholders)})
            """,
            params,
        )
    return {"ok": True}


async def list_favorites(user_id: int) -> list[dict[str, Any]]:
    rows = await db.fetch_all(
        """
        SELECT t.id, t.title, t.artists, t.album, t.duration_sec, t.isrc, t.cover_url,
               f.created_at AS liked_at,
               EXISTS (SELECT 1 FROM track_sources s
                       WHERE s.track_id = t.id AND s.connector = 'local') AS in_library
        FROM favorites f
        JOIN tracks t ON t.id = f.track_id
        WHERE f.user_id = :u
        ORDER BY f.created_at DESC
        """,
        {"u": user_id},
    )
    for r in rows:
        r["in_library"] = bool(r["in_library"])
    return rows


# ── Entity favorites: albums / artists (library cards, no downloads) ─────────


async def like_album(user_id: int, album_id: int) -> dict[str, Any]:
    album = await db.fetch_one("SELECT id FROM albums WHERE id = :a", {"a": album_id})
    if not album:
        return {"ok": False, "error": "album not found"}
    await db.execute(
        """
        INSERT INTO favorite_albums (user_id, album_id)
        VALUES (:u, :a)
        ON CONFLICT DO NOTHING
        """,
        {"u": user_id, "a": album_id},
    )
    return {"ok": True}


async def unlike_album(user_id: int, album_id: int) -> dict[str, Any]:
    await db.execute(
        "DELETE FROM favorite_albums WHERE user_id = :u AND album_id = :a",
        {"u": user_id, "a": album_id},
    )
    return {"ok": True}


async def like_artist(user_id: int, artist_id: int) -> dict[str, Any]:
    artist = await db.fetch_one("SELECT id FROM artists WHERE id = :a", {"a": artist_id})
    if not artist:
        return {"ok": False, "error": "artist not found"}
    await db.execute(
        """
        INSERT INTO favorite_artists (user_id, artist_id)
        VALUES (:u, :a)
        ON CONFLICT DO NOTHING
        """,
        {"u": user_id, "a": artist_id},
    )
    return {"ok": True}


async def unlike_artist(user_id: int, artist_id: int) -> dict[str, Any]:
    await db.execute(
        "DELETE FROM favorite_artists WHERE user_id = :u AND artist_id = :a",
        {"u": user_id, "a": artist_id},
    )
    # The cached discography is no longer pinned; let it expire and refresh.
    await db.execute(
        f"UPDATE artist_sync_state SET expires_at = NOW() + INTERVAL '{int(settings.artist_sync_ttl_days)} days' "
        "WHERE artist_id = :a AND expires_at IS NULL",
        {"a": artist_id},
    )
    return {"ok": True}
