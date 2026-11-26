"""Artists service: listing, artist pages (albums + top tracks)."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from .. import db
from ..config import settings
from ..connectors import get_connector
from . import resolve
from .albums import album_dto, page
from .dedup import collapse_track_rows, normalize_artist, normalize_title_or_raw
from .dto import build_track_dtos
from .entity_upsert import upsert_album, upsert_artist
from .normalize import split_artists
from .track_upsert import upsert_track_metas_bulk

log = logging.getLogger("kwhale.artists")

_DETAIL_TTL_SECONDS = 60.0
_detail_cache: dict[int, tuple[float, list[dict[str, Any]], list[int]]] = {}
_latest_cache: dict[int, tuple[float, dict[str, Any] | None]] = {}


def clear_detail_cache() -> None:
    _detail_cache.clear()


async def _best_effort(coro, default, seconds: float = 2.0):
    """Never let a connector stall an artist page."""
    try:
        return await asyncio.wait_for(coro, timeout=seconds)
    except Exception as exc:  # noqa: BLE001
        log.warning("artist enrichment skipped: %s", exc)
        return default

IN_LIBRARY_ARTIST_SQL = """
    EXISTS (
        SELECT 1 FROM tracks t
        JOIN track_sources s ON s.track_id = t.id AND s.connector = 'local'
        WHERE EXISTS (
            SELECT 1 FROM unnest(t.artists) AS an
            WHERE artist_matches(an, ar.name)
        )
    )
"""


def artist_dto(
    row: dict[str, Any], in_library: bool, is_favorite: bool = False
) -> dict[str, Any]:
    return {
        "id": row["id"],
        "name": row["name"],
        "cover_url": row["cover_url"],
        "genre": row.get("genre"),
        "in_library": in_library,
        "is_favorite": is_favorite,
    }


async def favorite_artist_ids(user_id: int, artist_ids: list[int]) -> set[int]:
    """Artist ids explicitly saved to the user's library."""
    if not artist_ids:
        return set()
    placeholders = [f":f{i}" for i in range(len(artist_ids))]
    params = {f"f{i}": aid for i, aid in enumerate(artist_ids)}
    params["u"] = user_id
    rows = await db.fetch_all(
        f"SELECT artist_id FROM favorite_artists "
        f"WHERE user_id = :u AND artist_id IN ({', '.join(placeholders)})",
        params,
    )
    return {int(r["artist_id"]) for r in rows}


async def artist_dto_by_ids(
    artist_ids: list[int], user_id: int | None = None, hide_empty: bool = False
) -> list[dict[str, Any]]:
    if not artist_ids:
        return []
    placeholders = [f":a{i}" for i in range(len(artist_ids))]
    params = {f"a{i}": aid for i, aid in enumerate(artist_ids)}
    empty_sql = (
        """
        , (
            ar.cover_url IS NOT NULL
            OR EXISTS (SELECT 1 FROM albums al WHERE al.artist_id = ar.id)
            OR EXISTS (SELECT 1 FROM track_artists ta WHERE ta.artist_id = ar.id)
            OR EXISTS (SELECT 1 FROM favorite_artists fa WHERE fa.artist_id = ar.id)
        ) AS has_content
        """
        if hide_empty
        else ""
    )
    rows = await db.fetch_all(
        f"SELECT id, name, cover_url, genre{empty_sql} FROM artists ar "
        f"WHERE id IN ({', '.join(placeholders)})",
        params,
    )
    if hide_empty:
        # Junk name-matches («бед бед бой…»): no cover, no tracks, no albums —
        # they only produce empty artist pages, so drop them when they are not
        # the only thing the query found.
        with_content = [r for r in rows if r.get("has_content")]
        rows = with_content or rows
    favorites = await favorite_artist_ids(user_id, artist_ids) if user_id else set()
    by_id = {r["id"]: r for r in rows}
    return [
        artist_dto(by_id[i], in_library=False, is_favorite=i in favorites)
        for i in artist_ids
        if i in by_id
    ]


async def list_artists(limit: int, offset: int) -> dict[str, Any]:
    total_row = await db.fetch_one("SELECT COUNT(*) AS n FROM artists")
    rows = await db.fetch_all(
        f"""
        SELECT ar.id, ar.name, ar.cover_url, ar.genre, {IN_LIBRARY_ARTIST_SQL} AS in_library
        FROM artists ar
        ORDER BY in_library DESC, lower(ar.name)
        LIMIT :l OFFSET :o
        """,
        {"l": limit, "o": offset},
    )
    items = [artist_dto(r, bool(r["in_library"])) for r in rows]
    return page(items, total_row["n"] if total_row else 0, limit, offset)


async def get_artist(
    artist_id: int, track_limit: int = 20, user_id: int | None = None
) -> dict[str, Any] | None:
    """Artist page: local-first, with a short, bounded network enrichment.

    The previous version awaited resolve/lazy connector calls without a
    timeout, so a slow provider made the whole screen hang or 500. Now the DB
    part is returned immediately; the network is best-effort and capped at 2s.
    """
    row = await db.fetch_one(
        f"""
        SELECT ar.*, {IN_LIBRARY_ARTIST_SQL} AS in_library
        FROM artists ar WHERE ar.id = :i
        """,
        {"i": artist_id},
    )
    if not row:
        return None

    now = time.monotonic()
    cached = _detail_cache.get(artist_id)
    if cached is not None and now - cached[0] < _DETAIL_TTL_SECONDS:
        album_items, all_track_ids = cached[1], cached[2]
    else:
        album_ids = await _artist_album_ids(row["name"])
        all_track_ids = await _db_artist_track_ids(artist_id, 1000)

        # One bounded enrichment pass only when the player has nothing local.
        if not album_ids or not all_track_ids:
            if not await _has_artist_sources(artist_id):
                meta = await _best_effort(resolve.find_artist_online(row["name"]), None)
                if meta is not None:
                    await upsert_artist(meta)
            if not album_ids:
                lazy_ids = await _best_effort(
                    _lazy_artist_albums(artist_id, row["name"]), []
                )
                album_ids = list(dict.fromkeys([*album_ids, *lazy_ids]))
            if not all_track_ids:
                lazy_tracks = await _best_effort(
                    _lazy_top_tracks(artist_id, 100), []
                )
                all_track_ids = list(dict.fromkeys([*all_track_ids, *lazy_tracks]))
            if all_track_ids:
                # Canonical DB order first. Anything the enrichment found that
                # the re-query does not return must survive — it used to be
                # silently wiped whenever the DB lookup came back short.
                db_ids = await _db_artist_track_ids(artist_id, 1000)
                all_track_ids = list(dict.fromkeys([*db_ids, *all_track_ids]))

        album_items = await _album_dtos(album_ids)
        _detail_cache[artist_id] = (now, album_items, all_track_ids)

    top_tracks = await build_track_dtos(all_track_ids[:track_limit], user_id)
    favorite = await db.fetch_one(
        "SELECT 1 FROM favorite_artists WHERE user_id = :u AND artist_id = :a",
        {"u": user_id, "a": artist_id},
    ) if user_id is not None else None
    dto = artist_dto(
        row, bool(row["in_library"]), is_favorite=favorite is not None
    )
    dto["album_count"] = len(album_items)
    latest_release = await _latest_release(artist_id, row["name"])
    await schedule_artist_sync_if_due(artist_id)
    return {
        "artist": dto,
        "albums": page(album_items, len(album_items), len(album_items), 0),
        "top_tracks": page(top_tracks, len(top_tracks), len(top_tracks), 0),
        "latest_release": latest_release,
    }


async def _db_artist_track_ids(artist_id: int, limit: int = 1000) -> list[int]:
    """All known tracks for an artist, ordered local/favorite/recent."""
    row = await db.fetch_one("SELECT name FROM artists WHERE id = :i", {"i": artist_id})
    if not row:
        return []
    # The pattern is normalized by the SAME SQL function that built the
    # tracks.artists_norm column. A Python-side key here once diverged from it
    # (homoglyph folding vs stock unaccent) and the prefilter matched nothing
    # for Cyrillic names — the «Тёмный Принц» bug (2026-11-18).
    # artists_norm is generated from artist_norm_imm(artists) and backed by a
    # GIN trigram index, so the LIKE is a candidate filter rather than a scan.
    # The EXISTS keeps artist_matches() token semantics (covers composite rows
    # and partial track_artists links, which a join alone would miss).
    rows = await db.fetch_all(
        """
        SELECT t.id, t.title, t.artists, t.duration_sec,
               EXISTS (SELECT 1 FROM track_sources s
                       WHERE s.track_id = t.id AND s.connector = 'local') AS local,
               EXISTS (SELECT 1 FROM favorites f WHERE f.track_id = t.id) AS fav
        FROM tracks t
        WHERE t.artists_norm LIKE '%' || artist_norm_imm(:name) || '%'
          AND EXISTS (
              SELECT 1 FROM unnest(t.artists) AS an
              WHERE artist_matches(an, :name)
          )
        ORDER BY local DESC, fav DESC, t.last_seen_at DESC
        LIMIT 5000
        """,
        {"name": row["name"]},
    )
    ids: list[int] = []
    for r in collapse_track_rows(rows):
        ids.append(r["id"])
        if len(ids) >= limit:
            break
    return ids


async def list_artist_tracks(
    artist_id: int, limit: int, offset: int, user_id: int | None = None
) -> dict[str, Any] | None:
    exists = await db.fetch_one("SELECT 1 FROM artists WHERE id = :i", {"i": artist_id})
    if not exists:
        return None
    await schedule_artist_sync_if_due(artist_id)
    ids = await _db_artist_track_ids(artist_id, 5000)
    chunk = ids[offset : offset + limit]
    return page(
        await build_track_dtos(chunk, user_id),
        len(ids),
        limit,
        offset,
    )


_sync_tasks: dict[int, asyncio.Task] = {}


async def artist_sync_status(artist_id: int) -> dict[str, Any]:
    row = await db.fetch_one(
        "SELECT * FROM artist_sync_state WHERE artist_id = :i", {"i": artist_id}
    )

    def _iso(value):
        return value.isoformat() if hasattr(value, "isoformat") else value

    if not row:
        return {
            "artistId": str(artist_id),
            "status": "idle",
            "trackCount": 0,
            "albumCount": 0,
            "error": None,
            "expiresAt": None,
        }
    return {
        "artistId": str(row["artist_id"]),
        "status": row["status"],
        "trackCount": row["track_count"],
        "albumCount": row["album_count"],
        "error": row.get("error"),
        "expiresAt": _iso(row.get("expires_at")),
    }


async def _needs_sync(artist_id: int) -> bool:
    row = await db.fetch_one(
        "SELECT * FROM artist_sync_state WHERE artist_id = :i", {"i": artist_id}
    )
    if row:
        started = row.get("started_at")
        status = row.get("status")
        if status == "running" and started is not None:
            age = (await db.fetch_one("SELECT NOW() - :t AS d", {"t": started}))["d"]
            if age.total_seconds() < 600:
                return False
        expires = row.get("expires_at")
        if expires is not None and expires > _now():
            return False
        if status == "done" and expires is None:
            return False
    # Automatic refresh only for artists that have a connector source; a
    # source-less artist is resolved explicitly by POST /sync.
    src = await db.fetch_one(
        """
        SELECT 1 FROM artist_sources
        WHERE artist_id = :i AND available
          AND external_id NOT LIKE 'album:%'
          AND external_id NOT LIKE 'track:%'
        LIMIT 1
        """,
        {"i": artist_id},
    )
    return src is not None


def _now():
    from datetime import UTC, datetime

    return datetime.now(UTC)


async def sync_artist(artist_id: int) -> None:
    """Persist full albums/tracks; liked artists never expire, others TTL."""
    await db.execute(
        """
        INSERT INTO artist_sync_state (artist_id, status, started_at, error)
        VALUES (:i, 'running', NOW(), NULL)
        ON CONFLICT (artist_id) DO UPDATE
        SET status = 'running', started_at = NOW(), error = NULL
        """,
        {"i": artist_id},
    )
    row = await db.fetch_one("SELECT id, name FROM artists WHERE id = :i", {"i": artist_id})
    if not row:
        return
    try:
        src = await db.fetch_one(
            """
            SELECT connector, external_id FROM artist_sources
            WHERE artist_id = :i AND available
              AND external_id NOT LIKE 'album:%'
              AND external_id NOT LIKE 'track:%'
            ORDER BY connector IN ('yandex', 'vk') DESC, connector = 'icm' DESC
            LIMIT 1
            """,
            {"i": artist_id},
        )
        if not src or src["connector"] not in ("yandex", "vk"):
            meta = await _best_effort(resolve.find_artist_online(row["name"]), None, 5.0)
            if meta is not None:
                await upsert_artist(meta)
            src = await db.fetch_one(
                """
                SELECT connector, external_id FROM artist_sources
                WHERE artist_id = :i AND available
                  AND external_id NOT LIKE 'album:%'
                  AND external_id NOT LIKE 'track:%'
                ORDER BY connector IN ('yandex', 'vk') DESC, connector = 'icm' DESC
                LIMIT 1
                """,
                {"i": artist_id},
            )
        if not src:
            await db.execute(
                """
                UPDATE artist_sync_state
                SET status = 'unavailable', finished_at = NOW(),
                    expires_at = NOW() + INTERVAL '6 hours', error = 'no source'
                WHERE artist_id = :i
                """,
                {"i": artist_id},
            )
            return

        connector = get_connector(src["connector"])
        if connector is None:
            raise RuntimeError(f"connector {src['connector']} not found")

        albums = await _best_effort(
            connector.artist_all_albums(src["external_id"], 200), [], 30.0
        )
        for meta in albums or []:
            await upsert_album(meta, artist_id=artist_id)
        tracks = await _best_effort(
            connector.artist_all_tracks(src["external_id"], 1000), [], 30.0
        )
        if tracks:
            await upsert_track_metas_bulk(tracks)

        all_track_ids = await _db_artist_track_ids(artist_id, 5000)
        album_ids = await _artist_album_ids(row["name"])
        liked = await db.fetch_one(
            "SELECT 1 FROM favorite_artists WHERE artist_id = :i LIMIT 1", {"i": artist_id}
        )
        expires_sql = "NULL" if liked else f"NOW() + INTERVAL '{int(settings.artist_sync_ttl_days)} days'"
        await db.execute(
            f"""
            UPDATE artist_sync_state
            SET status = 'done', finished_at = NOW(), expires_at = {expires_sql},
                track_count = :tc, album_count = :ac, error = NULL
            WHERE artist_id = :i
            """,
            {"i": artist_id, "tc": len(all_track_ids), "ac": len(album_ids)},
        )
        await db.execute("UPDATE artists SET last_seen_at = NOW() WHERE id = :i", {"i": artist_id})
        clear_detail_cache()
        _latest_cache.pop(artist_id, None)
        log.info(
            "artist %s synced: %s albums, %s tracks (expires %s)",
            artist_id,
            len(album_ids),
            len(all_track_ids),
            "never" if liked else f"{settings.artist_sync_ttl_days}d",
        )
    except Exception as exc:  # noqa: BLE001
        log.exception("artist %s sync failed", artist_id)
        await db.execute(
            """
            UPDATE artist_sync_state
            SET status = 'failed', finished_at = NOW(),
                expires_at = NOW() + INTERVAL '1 hour', error = :e
            WHERE artist_id = :i
            """,
            {"i": artist_id, "e": str(exc)[:500]},
        )


def schedule_artist_sync(artist_id: int) -> None:
    task = _sync_tasks.get(artist_id)
    if task is not None and not task.done():
        return
    task = asyncio.create_task(sync_artist(artist_id))
    _sync_tasks[artist_id] = task
    task.add_done_callback(lambda _t: _sync_tasks.pop(artist_id, None))


async def schedule_artist_sync_if_due(artist_id: int) -> None:
    if await _needs_sync(artist_id):
        schedule_artist_sync(artist_id)


async def _latest_release(artist_id: int, artist_name: str) -> dict[str, Any] | None:
    cached = _latest_cache.get(artist_id)
    if cached is not None and time.monotonic() - cached[0] < _DETAIL_TTL_SECONDS:
        return cached[1]
    ids = await _artist_album_ids(artist_name)
    if not ids:
        return None
    placeholders = [f":a{i}" for i in range(len(ids))]
    params = {f"a{i}": aid for i, aid in enumerate(ids)}
    params["ar"] = artist_name
    row = await db.fetch_one(
        f"""
        SELECT a.id, a.title, a.artist_id, a.year, a.release_date, a.track_count,
               a.cover_url, a.explicit, a.album_type, ar.name AS artist_name
        FROM albums a JOIN artists ar ON ar.id = a.artist_id
        WHERE a.id IN ({", ".join(placeholders)})
        ORDER BY a.year DESC NULLS LAST, a.release_date DESC NULLS LAST, a.id DESC
        LIMIT 1
        """,
        params,
    )
    if not row:
        return None
    dto = album_dto(
        row, in_library=False, artist_name=row["artist_name"], is_favorite=False
    )
    if not dto.get("release_date"):
        # Do not block the artist card on a cross-source date hunt. Fill it
        # in the background; the next open picks up the exact date.
        schedule_latest_release_enrichment(
            artist_id, row["title"], row["artist_name"]
        )
    _latest_cache[artist_id] = (time.monotonic(), dto)
    return dto


async def enrich_latest_release_date(artist_id: int, title: str, artist_name: str) -> None:
    meta = await _best_effort(resolve.find_album_online(title, artist_name), None, 4.0)
    date_value = meta.release_date if meta is not None else None
    if not date_value:
        sources = await db.fetch_all(
            """
            SELECT connector, external_id FROM artist_sources
            WHERE artist_id = :i AND available
              AND external_id NOT LIKE 'album:%'
              AND external_id NOT LIKE 'track:%'
            ORDER BY connector IN ('yandex', 'vk') DESC, connector = 'icm' DESC
            LIMIT 3
            """,
            {"i": artist_id},
        )
        wanted = normalize_title_or_raw(title)
        for src in sources:
            connector = get_connector(src["connector"])
            if connector is None:
                continue
            albums = await _best_effort(
                connector.artist_all_albums(src["external_id"], 200), [], 4.0
            )
            match = next(
                (
                    item
                    for item in albums or []
                    if normalize_title_or_raw(item.title) == wanted and item.release_date
                ),
                None,
            )
            if match is not None:
                date_value = match.release_date
                break
    if not date_value:
        return
    try:
        from datetime import date

        parsed = date.fromisoformat(str(date_value)[:10])
    except ValueError:
        return
    await db.execute(
        "UPDATE albums SET release_date = :d WHERE artist_id = :a AND lower(title) = lower(:t)",
        {"d": parsed, "a": artist_id, "t": title},
    )
    _latest_cache.pop(artist_id, None)


_latest_tasks: dict[int, asyncio.Task] = {}


def schedule_latest_release_enrichment(artist_id: int, title: str, artist_name: str) -> None:
    task = _latest_tasks.get(artist_id)
    if task is not None and not task.done():
        return
    task = asyncio.create_task(enrich_latest_release_date(artist_id, title, artist_name))
    _latest_tasks[artist_id] = task
    task.add_done_callback(lambda _t: _latest_tasks.pop(artist_id, None))



async def _album_dtos(album_ids: list[int]) -> list[dict[str, Any]]:
    if not album_ids:
        return []
    from .normalize import album_title_key

    placeholders = [f":a{i}" for i in range(len(album_ids))]
    params = {f"a{i}": aid for i, aid in enumerate(album_ids)}
    rows = await db.fetch_all(
        f"""
        SELECT a.id, a.title, a.artist_id, a.year, a.release_date, a.track_count, a.cover_url,
               a.explicit, a.album_type, ar.name AS artist_name
        FROM albums a JOIN artists ar ON ar.id = a.artist_id
        WHERE a.id IN ({", ".join(placeholders)})
        ORDER BY a.year DESC NULLS LAST, lower(a.title)
        """,
        params,
    )
    # Collapse residual duplicate DB rows ("WILD EA$T" + "WILD EA$T - Single",
    # cross-connector ±1 year release drift) keeping the richest row.
    deduped: list[dict[str, Any]] = []
    for r in rows:
        tkey = album_title_key(r["title"])
        ryear = r.get("year")
        matched_idx: int | None = None
        for idx, prev in enumerate(deduped):
            if album_title_key(prev["title"]) != tkey:
                continue
            pyear = prev.get("year")
            if pyear is None or ryear is None or abs(int(pyear) - int(ryear)) <= 1:
                matched_idx = idx
                break
        if matched_idx is None:
            deduped.append(r)
            continue
        prev = deduped[matched_idx]
        prev_rank = (1 if prev.get("cover_url") else 0, int(prev.get("track_count") or 0), -int(prev["id"]))
        cand_rank = (1 if r.get("cover_url") else 0, int(r.get("track_count") or 0), -int(r["id"]))
        if cand_rank > prev_rank:
            deduped[matched_idx] = r
    return [album_dto(r, in_library=False, artist_name=r["artist_name"]) for r in deduped]


async def list_favorite_artists(user_id: int, limit: int, offset: int) -> dict[str, Any]:
    """Liked artists (library cards), most recently saved first."""
    total_row = await db.fetch_one(
        "SELECT COUNT(*) AS n FROM favorite_artists WHERE user_id = :u", {"u": user_id}
    )
    rows = await db.fetch_all(
        f"""
        SELECT ar.id, ar.name, ar.cover_url, ar.genre, {IN_LIBRARY_ARTIST_SQL} AS in_library
        FROM favorite_artists fa
        JOIN artists ar ON ar.id = fa.artist_id
        WHERE fa.user_id = :u
        ORDER BY fa.created_at DESC
        LIMIT :l OFFSET :o
        """,
        {"u": user_id, "l": limit, "o": offset},
    )
    items = [
        artist_dto(r, bool(r["in_library"]), is_favorite=True) for r in rows
    ]
    return page(items, total_row["n"] if total_row else 0, limit, offset)


async def _has_artist_sources(artist_id: int) -> bool:
    row = await db.fetch_one(
        "SELECT 1 FROM artist_sources WHERE artist_id = :i AND available LIMIT 1",
        {"i": artist_id},
    )
    return row is not None


def _match_artist_ids(rows: list[dict[str, Any]], norm: str) -> list[int]:
    """Split-part match for legacy composite rows, off the event loop."""
    return [
        r["id"]
        for r in rows
        if norm in {normalize_artist(part) for part in (split_artists(r["name"]) or [r["name"]])}
    ]


async def _artist_album_ids(name: str) -> list[int]:
    norm = normalize_artist(name)
    if not norm:
        return []
    # The indexed exact match covers normal artists; the old full scan is only
    # a fallback for composite rows ("A & B" opened as "A"). It used to run on
    # every artist page and cost ~650 ms.
    rows = await db.fetch_all(
        "SELECT id FROM artists WHERE artist_norm_imm(name) = artist_norm_imm(:n)",
        {"n": name},
    )
    ids = [r["id"] for r in rows]
    if not ids:
        rows = await db.fetch_all("SELECT id, name FROM artists LIMIT 100000")
        ids = await asyncio.to_thread(_match_artist_ids, rows, norm)
    if not ids:
        return []
    placeholders = [f":i{i}" for i in range(len(ids))]
    params = {f"i{i}": v for i, v in enumerate(ids)}
    own = await db.fetch_all(
        f"SELECT id FROM albums WHERE artist_id IN ({', '.join(placeholders)})", params
    )
    return [r["id"] for r in own]


async def get_artist_albums(artist_id: int) -> dict[str, Any] | None:
    row = await db.fetch_one("SELECT id, name FROM artists WHERE id = :i", {"i": artist_id})
    if not row:
        return None
    own_ids = await _artist_album_ids(row["name"])
    lazy_ids = await _best_effort(_lazy_artist_albums(artist_id, row["name"]), [])
    await schedule_artist_sync_if_due(artist_id)
    items = await _album_dtos(list(dict.fromkeys([*own_ids, *lazy_ids])))
    return {"albums": page(items, len(items), len(items), 0)}


async def _lazy_artist_albums(artist_id: int, name: str) -> list[int]:
    """Merge albums from connector lazy fetch into DB; return new ids."""
    from ..connectors import get_connector
    from .entity_upsert import upsert_album

    src = await db.fetch_one(
        """
        SELECT connector, external_id FROM artist_sources
        WHERE artist_id = :i AND available
          AND external_id NOT LIKE 'album:%'
          AND external_id NOT LIKE 'track:%'
        ORDER BY connector = 'icm' DESC
        LIMIT 1
        """,
        {"i": artist_id},
    )
    if not src:
        return []
    connector = get_connector(src["connector"])
    if connector is None:
        return []
    metas = await connector.artist_albums(src["external_id"]) or []
    existing_ids = await _artist_album_ids(name)
    existing_norm: set[str] = set()
    if existing_ids:
        placeholders = [f":e{i}" for i in range(len(existing_ids))]
        params = {f"e{i}": aid for i, aid in enumerate(existing_ids)}
        ex_rows = await db.fetch_all(
            f"SELECT title FROM albums WHERE id IN ({', '.join(placeholders)})", params
        )
        existing_norm = {normalize_title_or_raw(r.get("title") or "") for r in ex_rows}
    new_ids: list[int] = []
    for m in metas:
        if normalize_title_or_raw(m.title) in existing_norm:
            continue
        new_ids.append(await upsert_album(m, artist_id=artist_id))
    return new_ids


async def _lazy_top_tracks(artist_id: int, limit: int = 20) -> list[int]:
    """Top tracks: prefer DB tracks by this artist, else lazy connector fetch.

    Local/favorite copies come first; duplicate rows for the same recording
    (historical upserts) are collapsed to one.
    """
    row = await db.fetch_one("SELECT name FROM artists WHERE id = :i", {"i": artist_id})
    if not row:
        return []
    norm = normalize_artist(row["name"]).replace(" ", "")
    db_rows = await db.fetch_all(
        """
        SELECT t.id, t.title, t.artists, t.duration_sec,
               EXISTS (SELECT 1 FROM track_sources s
                       WHERE s.track_id = t.id AND s.connector = 'local') AS local,
               EXISTS (SELECT 1 FROM favorites f WHERE f.track_id = t.id) AS fav
        FROM tracks t
        WHERE EXISTS (
            SELECT 1 FROM unnest(t.artists) AS an
            WHERE artist_matches(an, :ar)
        )
        ORDER BY
            local DESC,
            fav DESC,
            t.last_seen_at DESC
        LIMIT 100
        """,
        {"ar": norm},
    )
    if db_rows:
        ids: list[int] = []
        for r in collapse_track_rows(db_rows):
            ids.append(r["id"])
            if len(ids) >= limit:
                break
        return ids
    src = await db.fetch_one(
        """
        SELECT connector, external_id FROM artist_sources
        WHERE artist_id = :i AND available
          AND external_id NOT LIKE 'album:%'
          AND external_id NOT LIKE 'track:%'
        ORDER BY connector = 'icm' DESC
        LIMIT 1
        """,
        {"i": artist_id},
    )
    if not src:
        return []
    connector = get_connector(src["connector"])
    if connector is None:
        return []
    metas = await connector.artist_top_tracks(src["external_id"], limit) or []
    return await upsert_track_metas_bulk(metas)
