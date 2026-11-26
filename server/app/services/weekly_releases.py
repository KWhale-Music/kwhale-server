"""Friday 'new releases' playlist for liked artists.

One playlist per user, rebuilt in place every Friday from releases whose
release_date falls into the last seven days. No weekly playlist spam.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import date, timedelta

from .. import db
from . import albums as albums_service
from . import playlists as playlists_service

log = logging.getLogger("kwhale.weekly")

PLAYLIST_NAME = "Новинки пятницы"
MAX_TRACKS = 100


async def _album_track_ids(album_id: int, user_id: int) -> list[int]:
    rows = await db.fetch_all(
        """
        SELECT t.id
        FROM tracks t
        JOIN albums a ON a.id = :a
        JOIN artists ar ON ar.id = a.artist_id
        WHERE lower(regexp_replace(t.album, '\\s+', '', 'g'))
            = lower(regexp_replace(a.title, '\\s+', '', 'g'))
          AND EXISTS (
              SELECT 1 FROM unnest(t.artists) AS an(name)
              WHERE artist_matches(an.name, ar.name)
          )
        LIMIT 20
        """,
        {"a": album_id},
    )
    if rows:
        return [int(row["id"]) for row in rows]
    detail = await asyncio.wait_for(
        albums_service.get_album(album_id, user_id), timeout=6.0
    )
    if not detail:
        return []
    items = (detail.get("tracks") or {}).get("items") or []
    return [int(item["id"]) for item in items][:20]


async def _run_user(user_id: int, cutoff: date, today: date) -> int:
    release_rows = await db.fetch_all(
        """
        SELECT a.id
        FROM albums a
        JOIN artists ar ON ar.id = a.artist_id
        WHERE a.release_date IS NOT NULL
          AND a.release_date >= :cutoff
          AND a.release_date <= :today
          AND EXISTS (
              SELECT 1
              FROM favorite_artists fa
              JOIN artists liked ON liked.id = fa.artist_id
              WHERE fa.user_id = :u
                AND artist_matches(ar.name, liked.name)
          )
        ORDER BY a.release_date DESC, a.id DESC
        LIMIT 50
        """,
        {"u": user_id, "cutoff": cutoff, "today": today},
    )

    track_ids: list[int] = []
    seen: set[int] = set()
    for row in release_rows:
        for track_id in await _album_track_ids(int(row["id"]), user_id):
            if track_id not in seen:
                seen.add(track_id)
                track_ids.append(track_id)
            if len(track_ids) >= MAX_TRACKS:
                break
        if len(track_ids) >= MAX_TRACKS:
            break

    state = await db.fetch_one(
        "SELECT playlist_id FROM weekly_release_playlists WHERE user_id = :u",
        {"u": user_id},
    )
    playlist_id = int(state["playlist_id"]) if state and state.get("playlist_id") else None
    playlist = (
        await playlists_service.get_meta(playlist_id, user_id) if playlist_id else None
    )
    if playlist is None:
        created = await playlists_service.create(
            user_id,
            PLAYLIST_NAME,
            "Релизы любимых артистов за последние 7 дней",
            visibility="private",
            is_system=True,
            source="recommendation",
        )
        playlist_id = int(created["id"])
    else:
        await db.execute(
            """
            UPDATE playlists SET
                name = :n,
                description = :d,
                is_system = TRUE,
                source = 'recommendation',
                updated_at = NOW()
            WHERE id = :p
            """,
            {
                "n": PLAYLIST_NAME,
                "d": f"Релизы любимых артистов за последние 7 дней · {today.isoformat()}",
                "p": playlist_id,
            },
        )

    await db.execute(
        "DELETE FROM playlist_tracks WHERE playlist_id = :p", {"p": playlist_id}
    )
    if track_ids:
        await playlists_service.add_tracks(playlist_id, user_id, track_ids, 0)
    await db.execute(
        """
        INSERT INTO weekly_release_playlists (user_id, playlist_id, updated_at)
        VALUES (:u, :p, NOW())
        ON CONFLICT (user_id) DO UPDATE
        SET playlist_id = EXCLUDED.playlist_id, updated_at = NOW()
        """,
        {"u": user_id, "p": playlist_id},
    )
    return len(track_ids)


async def run(force: bool = False) -> dict[str, int]:
    today = date.today()
    if not force and today.weekday() != 4:  # Friday
        return {"users": 0, "tracks": 0, "skipped": 1}
    users = await db.fetch_all(
        "SELECT DISTINCT user_id FROM favorite_artists ORDER BY user_id"
    )
    total_tracks = 0
    processed = 0
    for row in users:
        user_id = int(row["user_id"])
        state = await db.fetch_one(
            "SELECT updated_at FROM weekly_release_playlists WHERE user_id = :u",
            {"u": user_id},
        )
        if (
            state
            and state.get("updated_at") is not None
            and state["updated_at"].date() == today
            and not force
        ):
            continue
        try:
            total_tracks += await _run_user(user_id, today - timedelta(days=7), today)
            processed += 1
        except Exception:  # noqa: BLE001
            log.exception("weekly releases for user %s failed", user_id)
    if processed:
        log.info("weekly releases: users=%s tracks=%s", processed, total_tracks)
    return {"users": processed, "tracks": total_tracks, "skipped": 0}
