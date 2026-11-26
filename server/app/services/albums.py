"""Albums service: listing, in_library flags, lazy online tracklists.

Albums live in the albums table (entity graph); a library album is one whose
title matches a local track's album text field (normalized compare).
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from .. import db
from ..connectors import get_connector
from . import resolve
from .dedup import collapse_track_rows, normalize_artist, normalize_title_or_raw
from .dto import build_track_dtos, page
from .entity_upsert import link_album_source
from .track_upsert import upsert_track_metas_bulk

log = logging.getLogger("kwhale.albums")

# Resolved tracklists are cached briefly: an album page used to pay three
# sequential network round-trips on EVERY open (~5 s for a single). The cache
# is invalidated whenever a new source is linked or tracks are upserted.
_TRACKLIST_TTL_SECONDS = 600.0
_tracklist_cache: dict[int, tuple[float, list[int]]] = {}


def clear_tracklist_cache(album_id: int | None = None) -> None:
    if album_id is None:
        _tracklist_cache.clear()
    else:
        _tracklist_cache.pop(album_id, None)


async def _best_effort(coro, default, seconds: float = 2.0):
    """Never let a connector stall an album page."""
    try:
        return await asyncio.wait_for(coro, timeout=seconds)
    except Exception as exc:  # noqa: BLE001
        log.warning("album enrichment skipped: %s", exc)
        return default


def _iso(value: Any) -> str | None:
    if value is None:
        return None
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def album_dto(
    row: dict[str, Any],
    in_library: bool,
    artist_name: str | None,
    is_favorite: bool = False,
    added_at: Any | None = None,
) -> dict[str, Any]:
    return {
        "id": row["id"],
        "title": row["title"],
        "artist": {"id": row["artist_id"], "name": artist_name},
        "year": row["year"],
        "release_date": (
            row.get("release_date").isoformat()
            if hasattr(row.get("release_date"), "isoformat")
            else row.get("release_date")
        ),
        "track_count": row["track_count"],
        "cover_url": row["cover_url"],
        "explicit": bool(row.get("explicit")),
        "album_type": row.get("album_type"),
        "in_library": in_library,
        "is_favorite": is_favorite,
        "added_at": _iso(added_at),
    }


async def favorite_album_ids(user_id: int, album_ids: list[int]) -> set[int]:
    """Album ids explicitly saved to the user's library."""
    if not album_ids:
        return set()
    placeholders = [f":f{i}" for i in range(len(album_ids))]
    params = {f"f{i}": aid for i, aid in enumerate(album_ids)}
    params["u"] = user_id
    rows = await db.fetch_all(
        f"SELECT album_id FROM favorite_albums "
        f"WHERE user_id = :u AND album_id IN ({', '.join(placeholders)})",
        params,
    )
    return {int(r["album_id"]) for r in rows}


async def _local_album_match_expr() -> str:
    """SQL fragment matching a local track by normalized album title."""
    return "lower(regexp_replace(t.album, '\\s+', '', 'g'))"


IN_LIBRARY_ALBUM_SQL = """
    EXISTS (
        SELECT 1 FROM tracks t
        JOIN track_sources s ON s.track_id = t.id AND s.connector = 'local'
        WHERE lower(regexp_replace(t.album, '\\s+', '', 'g'))
            = lower(regexp_replace(a.title, '\\s+', '', 'g'))
          AND EXISTS (
              SELECT 1 FROM unnest(t.artists) AS an(name)
              WHERE artist_matches(an.name, ar.name)
          )
    )
"""


async def album_dto_by_ids(
    album_ids: list[int], user_id: int | None = None
) -> list[dict[str, Any]]:
    """AlbumDTOs for search results (ordered, with artist names)."""
    if not album_ids:
        return []
    placeholders = [f":a{i}" for i in range(len(album_ids))]
    params = {f"a{i}": aid for i, aid in enumerate(album_ids)}
    rows = await db.fetch_all(
        f"""
        SELECT a.id, a.title, a.artist_id, a.year, a.release_date, a.track_count, a.cover_url,
               a.explicit, a.album_type, ar.name AS artist_name
        FROM albums a JOIN artists ar ON ar.id = a.artist_id
        WHERE a.id IN ({", ".join(placeholders)})
        """,
        params,
    )
    favorites = await favorite_album_ids(user_id, album_ids) if user_id else set()
    saved_at: dict[int, Any] = {}
    if user_id:
        fav_params = {f"f{i}": aid for i, aid in enumerate(album_ids)}
        fav_params["u"] = user_id
        fav_placeholders = [f":f{i}" for i in range(len(album_ids))]
        fav_rows = await db.fetch_all(
            f"SELECT album_id, created_at FROM favorite_albums "
            f"WHERE user_id = :u AND album_id IN ({', '.join(fav_placeholders)})",
            fav_params,
        )
        saved_at = {int(r["album_id"]): r["created_at"] for r in fav_rows}
    by_id = {r["id"]: r for r in rows}
    return [
        album_dto(
            by_id[i],
            in_library=False,
            artist_name=by_id[i]["artist_name"],
            is_favorite=i in favorites,
            added_at=saved_at.get(i),
        )
        for i in album_ids
        if i in by_id
    ]


async def list_albums(limit: int, offset: int) -> dict[str, Any]:
    """Browse listing: albums with local tracks first, then the rest."""
    total_row = await db.fetch_one("SELECT COUNT(*) AS n FROM albums")
    rows = await db.fetch_all(
        f"""
        SELECT a.id, a.title, a.artist_id, a.year, a.release_date, a.track_count, a.cover_url,
               a.explicit, a.album_type, ar.name AS artist_name, {IN_LIBRARY_ALBUM_SQL} AS in_library
        FROM albums a JOIN artists ar ON ar.id = a.artist_id
        ORDER BY in_library DESC, a.year DESC NULLS LAST, lower(a.title)
        LIMIT :l OFFSET :o
        """,
        {"l": limit, "o": offset},
    )
    items = [album_dto(r, bool(r["in_library"]), r["artist_name"]) for r in rows]
    return page(items, total_row["n"] if total_row else 0, limit, offset)


async def list_favorite_albums(user_id: int, limit: int, offset: int) -> dict[str, Any]:
    """Liked albums (library cards), most recently saved first."""
    total_row = await db.fetch_one(
        "SELECT COUNT(*) AS n FROM favorite_albums WHERE user_id = :u", {"u": user_id}
    )
    rows = await db.fetch_all(
        f"""
        SELECT a.id, a.title, a.artist_id, a.year, a.release_date, a.track_count, a.cover_url,
               a.explicit, a.album_type, ar.name AS artist_name, {IN_LIBRARY_ALBUM_SQL} AS in_library,
               fa.created_at AS added_at
        FROM favorite_albums fa
        JOIN albums a ON a.id = fa.album_id
        JOIN artists ar ON ar.id = a.artist_id
        WHERE fa.user_id = :u
        ORDER BY fa.created_at DESC
        LIMIT :l OFFSET :o
        """,
        {"u": user_id, "l": limit, "o": offset},
    )
    items = [
        album_dto(
            r,
            bool(r["in_library"]),
            r["artist_name"],
            is_favorite=True,
            added_at=r.get("added_at"),
        )
        for r in rows
    ]
    return page(items, total_row["n"] if total_row else 0, limit, offset)


async def get_album(album_id: int, user_id: int | None = None) -> dict[str, Any] | None:
    """Album page: the *full* tracklist from the network when possible.

    Preference order: online tracklist via existing sources -> resolve the
    album online by title+artist and link the source -> DB tracks matching the
    album text (offline fallback for entities with no connector presence).
    """
    row = await db.fetch_one(
        f"""
        SELECT a.*, ar.name AS artist_name, {IN_LIBRARY_ALBUM_SQL} AS in_library
        FROM albums a JOIN artists ar ON ar.id = a.artist_id
        WHERE a.id = :i
        """,
        {"i": album_id},
    )
    if not row:
        return None
    favorite = await db.fetch_one(
        "SELECT 1 FROM favorite_albums WHERE user_id = :u AND album_id = :a",
        {"u": user_id, "a": album_id},
    ) if user_id is not None else None
    dto = album_dto(
        row,
        bool(row["in_library"]),
        row["artist_name"],
        is_favorite=favorite is not None,
    )

    track_ids = await _album_track_ids_cached(album_id, row["title"], row["artist_name"])

    tracks = await build_track_dtos(track_ids, user_id)
    return {"album": dto, "tracks": page(tracks, len(tracks), len(tracks), 0)}


async def _album_track_ids_cached(album_id: int, title: str, artist_name: str) -> list[int]:
    """Tracklist for the album page: cached, network stages strictly bounded."""
    cached = _tracklist_cache.get(album_id)
    if cached is not None and time.monotonic() - cached[0] < _TRACKLIST_TTL_SECONDS:
        return cached[1]

    track_ids = await _best_effort(_lazy_online_tracklist(album_id), [], 3.0)
    if not track_ids:
        meta = await _best_effort(resolve.find_album_online(title, artist_name), None, 3.0)
        if meta is not None:
            await link_album_source(album_id, meta)
            clear_tracklist_cache(album_id)
            track_ids = await _best_effort(_lazy_online_tracklist(album_id), [], 3.0)
    if not track_ids:
        # symbol / provider-specific titles are not reachable by album search:
        # match inside the artist discography instead
        row = await db.fetch_one(
            "SELECT artist_id FROM albums WHERE id = :i", {"i": album_id}
        )
        if row and row["artist_id"]:
            linked = await _best_effort(
                resolve.link_album_via_artist(album_id, title, row["artist_id"], artist_name),
                False,
                3.0,
            )
            if linked:
                clear_tracklist_cache(album_id)
                track_ids = await _best_effort(_lazy_online_tracklist(album_id), [], 3.0)
    if not track_ids:
        track_ids = await _local_track_ids(title, artist_name)
    if track_ids:
        _tracklist_cache[album_id] = (time.monotonic(), track_ids)
    return track_ids


async def _local_track_ids(title: str, artist_name: str) -> list[int]:
    """Tracks of this album from the DB (offline fallback).

    Matching happens in Python with the same normalization used by dedup, so
    punctuation/symbol titles ('†', 'D.A.N.C.E.') still line up — SQL-side
    regexp_replace only strips whitespace and would not. Duplicate rows for the
    same recording are collapsed, preferring local/favorite copies.
    """
    norm_title = normalize_title_or_raw(title)
    norm_artist = normalize_artist(artist_name)
    if not norm_title or not norm_artist:
        return []
    rows = await db.fetch_all(
        """
        SELECT t.id, t.album, t.artists, t.title, t.track_number, t.duration_sec,
               EXISTS (SELECT 1 FROM track_sources s
                       WHERE s.track_id = t.id AND s.connector = 'local') AS local,
               EXISTS (SELECT 1 FROM favorites f WHERE f.track_id = t.id) AS fav
        FROM tracks t
        WHERE t.artists_text ILIKE :a
        LIMIT 1000
        """,
        {"a": f"%{artist_name}%"},
    )
    matched = [
        r
        for r in rows
        if r["album"]
        and normalize_title_or_raw(r["album"]) == norm_title
        and any(normalize_artist(a) == norm_artist for a in (r["artists"] or []))
    ]
    matched.sort(
        key=lambda r: (
            not r["local"],
            not r["fav"],
            r["track_number"] is None,
            r["track_number"] or 0,
            r["title"].lower(),
        )
    )
    out: list[int] = []
    for r in collapse_track_rows(matched):
        out.append(r["id"])
        if len(out) >= 200:
            break
    return out


async def _lazy_online_tracklist(album_id: int) -> list[int]:
    """Fetch the album's full tracklist from the best connector, upsert tracks."""
    src = await db.fetch_one(
        """
        SELECT connector, external_id FROM album_sources
        WHERE album_id = :i AND available
        ORDER BY connector = 'icm' DESC
        LIMIT 1
        """,
        {"i": album_id},
    )
    if not src:
        return []
    connector = get_connector(src["connector"])
    if connector is None:
        return []
    metas = await connector.album_tracks(src["external_id"])
    if not metas:
        return []
    return await upsert_track_metas_bulk(metas)
