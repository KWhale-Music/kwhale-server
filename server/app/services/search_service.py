"""Search service: fan-out connectors -> dedup -> upsert -> TrackDTO list.

Upsert rules:
  * ISRC conflict -> merge (update metadata, keep id)
  * same connector+external_id -> existing source row (refresh meta)
  * new source -> insert
Tracks get stable ids; repeated searches do not duplicate rows.
last_seen_at refreshed on every appearance (ephemeral GC depends on it).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from typing import Any

from .. import db
from ..config import settings
from ..connectors.base import TrackMeta
from .dedup import dedup_tracks
from .genres import normalize_genres
from .identity import identity_hash

log = logging.getLogger("kwhale.search")


async def cache_get(key: str, default: Any = None) -> Any | None:
    """TTL cache read (search_cache table). Used by connectors too (lazy album fetch)."""
    row = await db.fetch_one(
        "SELECT result_json FROM search_cache WHERE cache_key = :k AND expires_at > NOW()",
        {"k": key},
    )
    if row is None:
        return default
    try:
        return json.loads(row["result_json"])
    except (TypeError, ValueError):
        return row["result_json"]


async def cache_put(key: str, connector: str, value: Any, ttl_hours: int | None = None) -> None:
    """TTL cache write. ttl_hours=None -> default search cache TTL."""
    await db.execute(
        """
        INSERT INTO search_cache (cache_key, connector, result_json, expires_at)
        VALUES (:k, :c, :j, NOW() + make_interval(hours => :ttl))
        ON CONFLICT (cache_key) DO UPDATE
        SET result_json = EXCLUDED.result_json, expires_at = EXCLUDED.expires_at
        """,
        {
            "k": key,
            "c": connector,
            "j": json.dumps(value),
            "ttl": ttl_hours or settings.search_cache_ttl_hours,
        },
    )


async def _cache_get(key: str) -> Any | None:
    return await cache_get(key)


def _cache_key(connector: str, query: str, limit: int) -> str:
    raw = f"{connector}::{query}::{limit}"
    return hashlib.sha256(raw.encode()).hexdigest()


async def _active_connectors() -> dict[str, Any]:
    """Connectors allowed to fan out: the user's chosen (enabled) sources."""
    from ..connectors import get_connectors
    from .source_accounts import enabled_connectors

    enabled = await enabled_connectors()
    return {name: c for name, c in get_connectors().items() if name in enabled}


CONNECTOR_TIMEOUT_SEC = 4.5
# The client follows a partial quick response with wait=true, so the first
# wave only needs to fill the screen fast. 2.8 s pushed the worst quick call to
# 3.3 s; 1.5 s keeps the visible result under the 2 s budget and lets the full
# pass add the slow connectors.
FIRST_WAVE_TIMEOUT_SEC = 1.5
FIRST_WAVE_COALESCE_SEC = 0.15
# A second wave must never outlive the Android client's 20 s read timeout.
WAIT_ALL_TIMEOUT_SEC = 8.0
_INFLIGHT_BG_SEARCHES: dict[tuple[str, int], Any] = {}

# Process-local cache of completed search ids. A repeated query (the client
# fires quick + wait=true for the same text) then skips connectors and the
# persistence pass entirely. Only ids are cached: DTOs are rebuilt per user so
# likes/download flags stay fresh.
RESULT_CACHE_TTL_SEC = 600.0
_RESULT_CACHE_MAX = 200
_RESULT_CACHE: dict[tuple[str, int], tuple[float, dict[str, Any]]] = {}

# One writer at a time. The quick response and the background merge both
# persist overlapping tracks; running them concurrently deadlocked on
# track_sources/artists/albums (one search took 27 s while winning/losing
# locks). A single API process serves two users, so a queue is cheap.
_PERSIST_LOCK = asyncio.Lock()


def _is_deadlock(exc: BaseException) -> bool:
    orig = getattr(exc, "orig", None) or exc
    return getattr(orig, "sqlstate", None) == "40P01"


def _result_cache_get(key: tuple[str, int]) -> dict[str, Any] | None:
    import time

    item = _RESULT_CACHE.get(key)
    if item is None:
        return None
    expires, payload = item
    if expires < time.monotonic():
        _RESULT_CACHE.pop(key, None)
        return None
    return payload


def _result_cache_put(key: tuple[str, int], payload: dict[str, Any]) -> None:
    import time

    if len(_RESULT_CACHE) >= _RESULT_CACHE_MAX:
        oldest = min(_RESULT_CACHE, key=lambda k: _RESULT_CACHE[k][0])
        _RESULT_CACHE.pop(oldest, None)
    _RESULT_CACHE[key] = (time.monotonic() + RESULT_CACHE_TTL_SEC, payload)


async def _connector_search(name: str, connector, query: str, limit: int) -> list[TrackMeta]:
    """Search one connector through the TTL cache (metas serialised as dicts)."""
    import asyncio

    key = _cache_key(name, query, limit)
    cached = await _cache_get(key)
    if cached is not None:
        return [TrackMeta(**m) for m in cached]

    async def _fetch() -> list[TrackMeta]:
        if not await connector.is_available():
            return []
        return await connector.search(query, limit)

    try:
        metas = await asyncio.wait_for(_fetch(), timeout=CONNECTOR_TIMEOUT_SEC)
    except Exception as exc:
        log.warning("connector %s tracks search failed/timed out: %s", name, exc)
        return []

    serialisable = [
        {
            "connector": m.connector,
            "external_id": m.external_id,
            "title": m.title,
            "artists": m.artists,
            "album": m.album,
            "duration_sec": m.duration_sec,
            "isrc": m.isrc,
            "cover_url": m.cover_url,
            "quality_tier": m.quality_tier,
            "year": m.year,
            "track_number": m.track_number,
            "track_total": m.track_total,
            "disc_number": m.disc_number,
            "disc_total": m.disc_total,
            "explicit": m.explicit,
            "genre": m.genre,
        }
        for m in metas
    ]
    if serialisable:
        await cache_put(key, name, serialisable)
    return metas


async def _fan_out(query: str, limit: int) -> list[TrackMeta]:
    import asyncio

    connectors = await _active_connectors()
    tasks = [_connector_search(name, conn, query, limit) for name, conn in connectors.items()]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    metas: list[TrackMeta] = []
    for name, result in zip(connectors, results, strict=False):
        if isinstance(result, BaseException):
            log.warning("connector %s failed: %s", name, result)
            continue
        metas.extend(result)
    return metas


async def _canonical_meta(sources: list[TrackMeta]) -> TrackMeta | None:
    """Canonical non-local metadata: configured source priority, then quality, then explicit.

    Backfills missing fields (e.g. ICM search hits have duration_sec=0) and
    prefers unpacked participant lists over packed "A & B" strings from any source.
    """
    if not sources:
        return None
    from dataclasses import replace

    from ..v1.priority import metadata_order
    from .normalize import flatten_artist_names, split_artists_strict

    order = await metadata_order()
    rank = {name: idx for idx, name in enumerate(order)}
    best = min(
        sources,
        key=lambda m: (
            rank.get(m.connector, len(order) + 1),
            -int(m.quality_tier or 0),
            not bool(m.explicit),
            m.connector,
        ),
    )
    dur = best.duration_sec or next((s.duration_sec for s in sources if s.duration_sec), 0)
    isrc = best.isrc or next((s.isrc for s in sources if s.isrc), None)
    cover = best.cover_url or next((s.cover_url for s in sources if s.cover_url), "")
    year = best.year or next((s.year for s in sources if s.year), None)
    album = best.album or next((s.album for s in sources if s.album), "")
    artists = list(best.artists or [])
    # Prefer the richest properly-split collaborator array across sources
    best_flat = flatten_artist_names(artists)
    split_candidates = [
        list(s.artists or [])
        for s in sources
        if len(s.artists or []) >= 2
        or (len(best.artists or []) == 1 and len(split_artists_strict(best.artists[0])) > 1)
    ]
    if split_candidates:
        richest = max(split_candidates, key=lambda lst: len(flatten_artist_names(lst)))
        if len(richest) > len(artists) or (len(artists) == 1 and len(best_flat) > 1):
            artists = [p for a in richest for p in (split_artists_strict(a) or [a]) if p]
    explicit = True if any(s.explicit is True for s in sources) else best.explicit
    return replace(
        best,
        artists=artists,
        album=album,
        duration_sec=dur,
        isrc=isrc,
        cover_url=cover,
        year=year,
        explicit=explicit,
    )


_SOURCE_META_SQL = """
INSERT INTO track_source_metadata
    (track_id, connector, external_id, title, artists, album, year,
     track_number, track_total, disc_number, disc_total, genre,
     cover_url, isrc, duration_sec, quality_tier, updated_at)
VALUES
    (:tid, :conn, :ext, :title, :artists, :album, :year,
     :tn, :tt, :dn, :dt, :genre, :cover, :isrc, :duration, :tier, NOW())
ON CONFLICT (track_id, connector, external_id) DO UPDATE SET
    title = EXCLUDED.title,
    artists = EXCLUDED.artists,
    album = EXCLUDED.album,
    year = EXCLUDED.year,
    track_number = EXCLUDED.track_number,
    track_total = EXCLUDED.track_total,
    disc_number = EXCLUDED.disc_number,
    disc_total = EXCLUDED.disc_total,
    genre = EXCLUDED.genre,
    cover_url = EXCLUDED.cover_url,
    isrc = EXCLUDED.isrc,
    duration_sec = EXCLUDED.duration_sec,
    quality_tier = EXCLUDED.quality_tier,
    updated_at = NOW()
"""

_TRACK_SOURCE_SQL = """
INSERT INTO track_sources
    (track_id, connector, external_id, quality_tier, mime, available, updated_at)
VALUES (:tid, :conn, :ext, :tier, NULL, TRUE, NOW())
ON CONFLICT (connector, external_id) DO UPDATE
SET quality_tier = EXCLUDED.quality_tier,
    available = TRUE,
    track_id = EXCLUDED.track_id,
    updated_at = NOW()
"""


def _source_meta_row(track_id: int, m: TrackMeta) -> dict[str, Any]:
    return {
        "tid": track_id,
        "conn": m.connector,
        "ext": m.external_id,
        "title": m.title,
        "artists": m.artists,
        "album": m.album or None,
        "year": m.year,
        "tn": m.track_number,
        "tt": m.track_total,
        "dn": m.disc_number,
        "dt": m.disc_total,
        "genre": m.genre,
        "cover": m.cover_url or None,
        "isrc": m.isrc,
        "duration": m.duration_sec or None,
        "tier": m.quality_tier,
    }


async def _store_source_meta(track_id: int, m: TrackMeta) -> None:
    await db.execute_many(_SOURCE_META_SQL, [_source_meta_row(track_id, m)])


async def _upsert_track(
    meta: TrackMeta,
    sources: list[TrackMeta],
    *,
    source_rows: list[dict[str, Any]] | None = None,
    meta_rows: list[dict[str, Any]] | None = None,
    enqueue: bool = True,
) -> int:
    """Insert/merge a track, store per-source metadata, preserve local truth.

    Canonical metadata for non-local tracks comes from the configured source
    priority. Once a local file exists, remote searches never overwrite its
    canonical metadata; they only add/refresh source rows.

    A search passes `source_rows`/`meta_rows` so the caller can flush all
    per-source rows with two executemany calls instead of ~600 round trips.
    """
    existing = None
    if meta.isrc:
        existing = await db.fetch_one("SELECT id FROM tracks WHERE isrc = :i", {"i": meta.isrc})
    if existing is None:
        existing = await _fuzzy_existing(meta)

    track_id: int
    has_local = False
    if existing is not None:
        track_id = existing["id"]
        has_local = bool(
            await db.fetch_one(
                "SELECT 1 FROM track_sources WHERE track_id = :t AND connector = 'local'",
                {"t": track_id},
            )
        )
    canonical = meta if has_local else (await _canonical_meta(sources) or meta)
    artists = [a for a in (canonical.artists or []) if a] or ["Unknown"]
    ih = identity_hash(canonical.title, artists)
    gn = normalize_genres([canonical.genre]) or None

    if existing is None:
        if canonical.isrc:
            rows = await db.execute_returning(
                """
                INSERT INTO tracks
                    (title, artists, album, duration_sec, isrc, cover_url, year,
                     track_number, track_total, disc_number, disc_total, explicit,
                     genre, genres_norm, identity_hash, last_seen_at)
                VALUES (:t, :a, :al, :d, :i, :c, :y, :tn, :tt, :dn, :dt, :ex, :g,
                        COALESCE(CAST(:gn AS text[]), '{}'), :ih, NOW())
                ON CONFLICT (isrc) DO UPDATE SET last_seen_at = NOW(),
                    genres_norm = CASE WHEN tracks.genres_norm = '{}'
                        THEN EXCLUDED.genres_norm ELSE tracks.genres_norm END
                RETURNING id
                """,
                {
                    "t": canonical.title,
                    "a": artists,
                    "al": canonical.album or None,
                    "d": canonical.duration_sec or None,
                    "i": canonical.isrc,
                    "c": canonical.cover_url or None,
                    "y": canonical.year,
                    "tn": canonical.track_number,
                    "tt": canonical.track_total,
                    "dn": canonical.disc_number,
                    "dt": canonical.disc_total,
                    "ex": canonical.explicit,
                    "g": canonical.genre,
                    "gn": gn,
                    "ih": ih,
                },
            )
        else:
            rows = await db.execute_returning(
                """
                INSERT INTO tracks
                    (title, artists, album, duration_sec, isrc, cover_url, year,
                     track_number, track_total, disc_number, disc_total, explicit,
                     genre, genres_norm, identity_hash, last_seen_at)
                VALUES (:t, :a, :al, :d, NULL, :c, :y, :tn, :tt, :dn, :dt, :ex, :g,
                        COALESCE(CAST(:gn AS text[]), '{}'), :ih, NOW())
                RETURNING id
                """,
                {
                    "t": canonical.title,
                    "a": artists,
                    "al": canonical.album or None,
                    "d": canonical.duration_sec or None,
                    "c": canonical.cover_url or None,
                    "y": canonical.year,
                    "tn": canonical.track_number,
                    "tt": canonical.track_total,
                    "dn": canonical.disc_number,
                    "dt": canonical.disc_total,
                    "ex": canonical.explicit,
                    "g": canonical.genre,
                    "gn": gn,
                    "ih": ih,
                },
            )
        track_id = rows[0]["id"]
    elif has_local:
        # Local file is the source of truth; just refresh the ephemeral TTL.
        await db.execute("UPDATE tracks SET last_seen_at = NOW() WHERE id = :id", {"id": track_id})
    else:
        await db.execute(
            """
            UPDATE tracks SET
                title = COALESCE(NULLIF(:t, ''), title),
                artists = :a,
                duration_sec = COALESCE(:d, duration_sec),
                cover_url = COALESCE(:c, cover_url),
                album = COALESCE(:al, album),
                year = COALESCE(:y, year),
                track_number = COALESCE(:tn, track_number),
                track_total = COALESCE(:tt, track_total),
                disc_number = COALESCE(:dn, disc_number),
                disc_total = COALESCE(:dt, disc_total),
                explicit = COALESCE(:ex, explicit),
                genre = COALESCE(:g, genre),
                genres_norm = COALESCE(CAST(:gn AS text[]), genres_norm),
                identity_hash = COALESCE(:ih, identity_hash),
                last_seen_at = NOW()
            WHERE id = :id
            """,
            {
                "t": canonical.title,
                "a": artists,
                "d": canonical.duration_sec or None,
                "c": canonical.cover_url or None,
                "al": canonical.album or None,
                "y": canonical.year,
                "tn": canonical.track_number,
                "tt": canonical.track_total,
                "dn": canonical.disc_number,
                "dt": canonical.disc_total,
                "ex": canonical.explicit,
                "g": canonical.genre,
                "gn": gn,
                "ih": ih,
                "id": track_id,
            },
        )

    for s in sources:
        row = {
            "tid": track_id,
            "conn": s.connector,
            "ext": s.external_id,
            "tier": s.quality_tier,
        }
        if source_rows is None:
            await db.execute(_TRACK_SOURCE_SQL, row)
        else:
            source_rows.append(row)
        meta_row = _source_meta_row(track_id, s)
        if meta_rows is None:
            await _store_source_meta(track_id, s)
        else:
            meta_rows.append(meta_row)
    if enqueue:
        from .rec.queue import enqueue_track

        await enqueue_track(track_id, kinds=("text",))
    return track_id


async def _fuzzy_existing(meta: TrackMeta) -> dict[str, Any] | None:
    """Find an existing row for the same recording.

    Identity comes from services.identity: (title, primary artist after strict
    collab splitting) with a ±15 s duration window — the same rule merge.py
    uses to consolidate rows. The old ±2 s / first-artist-text match was
    narrower than the merge rule, so every search kept re-creating duplicates
    that repair had already collapsed: the source of the «лайк не виден /
    дубли в библиотеке» bug. When the identity hash is missing (pre-backfill
    rows) we compute the same key on the fly over a small candidate set.
    """
    from .dedup import normalize_title
    from .identity import DURATION_TOLERANCE_SEC, duration_close, identity_hash

    title = normalize_title(meta.title)
    if not title:
        return None
    ih = identity_hash(meta.title, meta.artists)

    rows = await db.fetch_all(
        "SELECT id, title, artists, duration_sec, isrc, identity_hash FROM tracks "
        "WHERE identity_hash = :ih LIMIT 200",
        {"ih": ih},
    )
    for r in rows:
        if duration_close(r["duration_sec"] or 0, meta.duration_sec, DURATION_TOLERANCE_SEC):
            return r

    # Legacy / collaborator-packed rows without matching hash: compute full
    # same_recording compatibility over the trgm candidate window.
    from .identity import same_recording

    raw_first = meta.artists[0] if meta.artists else ""
    candidates = await db.fetch_all(
        """
        SELECT id, title, artists, duration_sec, isrc FROM tracks
        WHERE artists_text ILIKE :a
           OR artists_norm LIKE '%' || artist_norm_imm(:a_raw) || '%'
        LIMIT 500
        """,
        {"a": f"%{raw_first}%", "a_raw": raw_first},
    )
    for r in candidates:
        if same_recording(
            r["title"], r["artists"], r["duration_sec"],
            meta.title, meta.artists, meta.duration_sec,
        ):
            return r
    return None


async def _track_dto(track_id: int) -> dict[str, Any]:
    track = await db.fetch_one("SELECT * FROM tracks WHERE id = :id", {"id": track_id})
    if track is None:
        raise ValueError(f"track {track_id} vanished")
    sources = await db.fetch_all(
        "SELECT connector, external_id, quality_tier, file_path, mime "
        "FROM track_sources WHERE track_id = :id ORDER BY quality_tier DESC",
        {"id": track_id},
    )
    return {
        "id": track_id,
        "title": track["title"],
        "artists": track["artists"],
        "album": track["album"],
        "duration_sec": track["duration_sec"],
        "isrc": track["isrc"],
        "cover_url": track["cover_url"],
        "in_library": any(s["connector"] == "local" for s in sources),
        "sources": [
            {
                "connector": s["connector"],
                "external_id": s["external_id"],
                "quality_tier": s["quality_tier"],
            }
            for s in sources
        ],
    }


async def _persist_and_format(
    query: str,
    limit: int,
    user_id: int | None,
    track_metas: list[TrackMeta],
    album_metas: list[Any],
    artist_metas: list[Any],
    errors: list[str],
    partial: bool = False,
    with_dtos: bool = True,
) -> dict[str, Any]:
    """Persist and format one search in a single DB transaction.

    Per-row autocommit used to fsync ~1,400 times per search (~11 s). The
    wrapper joins every helper below into one transaction; savepoints inside
    keep one bad track/album/artist from aborting the whole batch.
    `with_dtos=False` returns ids only — the caller builds DTOs once for the
    whole response instead of paying for them twice.
    """
    async with _PERSIST_LOCK:
        async with db.batch():
            return await _persist_and_format_inner(
                query,
                limit,
                user_id,
                track_metas,
                album_metas,
                artist_metas,
                errors,
                partial,
                with_dtos,
            )


async def _persist_and_format_inner(
    query: str,
    limit: int,
    user_id: int | None,
    track_metas: list[TrackMeta],
    album_metas: list[Any],
    artist_metas: list[Any],
    errors: list[str],
    partial: bool = False,
    with_dtos: bool = True,
) -> dict[str, Any]:
    from ..v1.priority import search_rank
    from .dedup import dedup_albums, dedup_artists
    from .entity_upsert import upsert_album, upsert_artist
    from .search_rank import match_score

    # Relevance decides what is worth persisting. Source/quality is only a
    # tie breaker now; the old order made ICM rows win regardless of the query.
    track_groups = dedup_tracks(track_metas)
    track_groups.sort(
        key=lambda g: (
            -match_score(query, g["winner"].title, *g["winner"].artists),
            *search_rank(
                getattr(g["winner"], "connector", ""),
                getattr(g["winner"], "quality_tier", 0) or 0,
            ),
        )
    )
    track_ids: list[int] = []
    source_rows: list[dict[str, Any]] = []
    meta_rows: list[dict[str, Any]] = []
    # Persist a little more than the response limit so the post-upsert
    # popularity pass can reorder without losing candidates.
    for g in track_groups[: max(limit * 2, limit)]:
        for attempt in range(2):
            try:
                async with db.savepoint():
                    track_ids.append(
                        await _upsert_track(
                            g["winner"],
                            g["sources"],
                            source_rows=source_rows,
                            meta_rows=meta_rows,
                            enqueue=False,
                        )
                    )
                break
            except Exception as e:
                if attempt == 0 and _is_deadlock(e):
                    # The downloader/indexer can still race us on shared rows.
                    await asyncio.sleep(0.05)
                    continue
                log.exception("upsert failed for %r: %s", g["winner"].title, e)
                errors.append(str(e))
                break

    # Two bulk statements instead of one round trip per source row (~600).
    # PostgreSQL rejects a second ON CONFLICT hit for the same key in one
    # statement, so collapse duplicates first.
    try:
        if source_rows:
            unique_sources = {(r["conn"], r["ext"]): r for r in source_rows}
            async with db.savepoint():
                await db.execute_many(_TRACK_SOURCE_SQL, list(unique_sources.values()))
        if meta_rows:
            unique_meta = {(r["tid"], r["conn"], r["ext"]): r for r in meta_rows}
            async with db.savepoint():
                await db.execute_many(_SOURCE_META_SQL, list(unique_meta.values()))
    except Exception as e:  # noqa: BLE001
        log.exception("source rows flush failed: %s", e)
        errors.append(str(e))

    if track_ids:
        from .rec.queue import MODEL_VERSIONS

        await db.execute(
            """
            INSERT INTO index_jobs (track_id, kind, model_version)
            SELECT unnest(CAST(:ids AS bigint[])), 'text', :v
            ON CONFLICT (track_id, kind, model_version) DO UPDATE
            SET status = 'pending', attempts = 0, last_error = NULL, updated_at = NOW()
            WHERE index_jobs.status = 'failed'
            """,
            {"ids": list(dict.fromkeys(track_ids)), "v": MODEL_VERSIONS["text"]},
        )

    album_groups = dedup_albums(album_metas)
    album_groups.sort(
        key=lambda g: (
            -match_score(query, g["winner"].title, g["winner"].artist_name),
            0 if g["winner"].connector == "icm" else 1,
        )
    )
    album_ids: list[int] = []
    for g in album_groups[: max(10, limit)]:
        try:
            async with db.savepoint():
                album_ids.append(await upsert_album(g["winner"]))
        except Exception as e:
            log.warning("album upsert failed for %r: %s", g["winner"].title, e)
            errors.append(str(e))

    artist_groups = dedup_artists(artist_metas)
    artist_groups.sort(
        key=lambda g: (
            -match_score(query, g["winner"].name),
            0 if g["winner"].connector == "icm" else 1,
        )
    )
    artist_ids: list[int] = []
    for g in artist_groups[: max(10, limit)]:
        try:
            async with db.savepoint():
                artist_ids.append(await upsert_artist(g["winner"]))
        except Exception as e:
            log.warning("artist upsert failed for %r: %s", g["winner"].name, e)
            errors.append(str(e))

    if with_dtos:
        from .albums import album_dto_by_ids
        from .artists import artist_dto_by_ids
        from .dto import build_track_dtos

        track_items = await build_track_dtos(track_ids, user_id)
        album_items = await album_dto_by_ids(album_ids)
        artist_items = await artist_dto_by_ids(artist_ids)
    else:
        # The API router rebuilds DTOs once for the final, local-merged order.
        track_items = [{"id": tid} for tid in track_ids]
        album_items = [{"id": aid} for aid in album_ids]
        artist_items = [{"id": aid} for aid in artist_ids]

    def _section_of(items: list[dict[str, Any]]) -> dict[str, Any]:
        return {"items": items, "errors": errors}

    return {
        "query": query,
        "tracks": _section_of(track_items),
        "albums": _section_of(album_items),
        "artists": _section_of(artist_items),
        "errors": errors,
        "partial": partial,
    }


async def search(
    query: str,
    limit: int,
    user_id: int | None = None,
    wait_all: bool = False,
    with_dtos: bool = True,
) -> dict[str, Any]:
    """Progressive two-wave pipeline:
    - wait_all=False (default): returns as soon as the first fast connector(s)
      deliver tracks (+ short coalesce window), while a background task finishes
      the remaining connectors and deduplicates/merges them into PostgreSQL.
    - wait_all=True: awaits all connectors (or any in-flight background merge)
      and returns the complete deduplicated result.
    """
    import asyncio

    inflight_key = (query.strip().lower(), limit)
    # The ids-only cache is consumed by the API router (with_dtos=False).
    # Callers asking for full DTOs keep the old always-fresh behaviour.
    if not with_dtos:
        cached = _result_cache_get(inflight_key)
        if cached is not None:
            return cached
    if wait_all:
        bg_task = _INFLIGHT_BG_SEARCHES.get(inflight_key)
        if bg_task is not None and not bg_task.done():
            try:
                await asyncio.wait_for(
                    asyncio.shield(bg_task), timeout=WAIT_ALL_TIMEOUT_SEC
                )
            except Exception:
                pass
        if not with_dtos:
            cached = _result_cache_get(inflight_key)
            if cached is not None:
                return cached

    connectors = await _active_connectors()
    task_entries: list[tuple[str, str, asyncio.Task]] = []
    for name, conn in connectors.items():
        task_entries.append(
            (name, "tracks", asyncio.create_task(_connector_search(name, conn, query, limit)))
        )
        task_entries.append(
            (
                name,
                "albums",
                asyncio.create_task(
                    _cached_section(name, conn, query, limit, "albums", conn.search_albums)
                ),
            )
        )
        task_entries.append(
            (
                name,
                "artists",
                asyncio.create_task(
                    _cached_section(name, conn, query, limit, "artists", conn.search_artists)
                ),
            )
        )

    if not task_entries:
        result = await _persist_and_format(
            query, limit, user_id, [], [], [], [], partial=False, with_dtos=with_dtos
        )
        if not with_dtos:
            _result_cache_put(inflight_key, result)
        return result

    if wait_all:
        _, pending = await asyncio.wait(
            {t for _, _, t in task_entries},
            timeout=WAIT_ALL_TIMEOUT_SEC,
        )
    else:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + FIRST_WAVE_TIMEOUT_SEC
        task_map = {t: (name, kind) for name, kind, t in task_entries}
        track_tasks = [t for _n, kind, t in task_entries if kind == "tracks"]
        pending = set(task_map.keys())
        got_first_tracks = False

        while pending:
            remaining = deadline - loop.time()
            if remaining <= 0:
                break
            done, pending = await asyncio.wait(
                pending,
                timeout=remaining,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if not done:
                break
            for d in done:
                _name, kind = task_map[d]
                if kind == "tracks" and not d.cancelled() and d.exception() is None:
                    if d.result():
                        got_first_tracks = True
            if got_first_tracks:
                if pending:
                    _, pending = await asyncio.wait(pending, timeout=FIRST_WAVE_COALESCE_SEC)
                break
            if all(t.done() for t in track_tasks):
                break

    track_metas: list[TrackMeta] = []
    album_metas: list[Any] = []
    artist_metas: list[Any] = []
    errors: list[str] = []
    for _name, kind, t in task_entries:
        if not t.done():
            continue
        if t.cancelled():
            continue
        exc = t.exception()
        if exc is not None:
            errors.append(f"{kind}: {exc}")
            continue
        res = t.result()
        if kind == "tracks":
            track_metas.extend(res)
        elif kind == "albums":
            album_metas.extend(res)
        else:
            artist_metas.extend(res)

    is_partial = bool(pending)
    if pending:
        # Do not leave connector tasks running after the response: their
        # parsing/cache writes competed with the next requests and made later
        # quick searches slower (1.7 s -> 3-4 s in one session). The wait=true
        # pass re-fetches the few connectors that were still in flight.
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
    if not wait_all:
        # Quick preview: persist at most one page. The wait=true pass merges
        # the complete union, so truncating here only bounds the preview write
        # (150 raw tracks from all connectors used to add ~1.5 s to the quick
        # response for no visible benefit).
        track_metas = track_metas[:limit]
        album_metas = album_metas[:limit]
        artist_metas = artist_metas[:limit]
    # No automatic background persist: the client immediately follows a partial
    # quick response with wait=true, and that full pass persists the union once.
    # The old fire-and-forget merge duplicated the same rows, and with a write
    # lock it queued behind new searches (one cold search took 29 s). Pending
    # connector tasks are left running so their results land in search_cache for
    # the wait=true pass.
    result = await _persist_and_format(
        query,
        limit,
        user_id,
        track_metas,
        album_metas,
        artist_metas,
        errors,
        partial=is_partial,
        with_dtos=with_dtos,
    )
    if not is_partial and not with_dtos:
        _result_cache_put(inflight_key, result)
    return result


async def _cached_section(name: str, conn, query: str, limit: int, kind: str, fn) -> list:
    """Albums/artists section for one connector through the TTL cache."""
    import asyncio

    from ..connectors.base import AlbumMeta, ArtistMeta

    key = _cache_key(f"{name}.{kind}", query, limit)
    cached = await _cache_get(key)
    if cached is not None:
        cls = AlbumMeta if kind == "albums" else ArtistMeta
        return [cls(**m) for m in cached]

    async def _fetch() -> list:
        if not await conn.is_available():
            return []
        return await fn(query, limit)

    try:
        metas = await asyncio.wait_for(_fetch(), timeout=CONNECTOR_TIMEOUT_SEC)
    except Exception as exc:
        log.warning("connector %s %s search failed/timed out: %s", name, kind, exc)
        return []
    serialisable = [
        {
            "connector": m.connector,
            "external_id": m.external_id,
            "title": m.title,
            "artist_name": getattr(m, "artist_name", ""),
            "year": getattr(m, "year", None),
            "cover_url": m.cover_url,
            "track_count": getattr(m, "track_count", None),
            "genre": getattr(m, "genre", None),
            "raw": {},
        }
        if kind == "albums"
        else {
            "connector": m.connector,
            "external_id": m.external_id,
            "name": m.name,
            "cover_url": m.cover_url,
            "genre": getattr(m, "genre", None),
            "raw": {},
        }
        for m in metas
    ]
    if serialisable:
        await cache_put(key, name, serialisable)
    return metas


async def upsert_track_metas(metas: list[TrackMeta]) -> list[int]:
    """Dedup+upsert arbitrary TrackMeta batch (lazy fetches), ids in order."""
    groups = dedup_tracks(metas)
    ids: list[int] = []
    for g in groups:
        try:
            ids.append(await _upsert_track(g["winner"], g["sources"]))
        except Exception as e:
            log.warning("track upsert %r: %s", g["winner"].title, e)
    return ids


async def search_local_ids(query: str, limit: int = 50) -> list[int]:
    """Local library first: downloaded/uploaded tracks matching the query."""
    q = f"%{query.strip()}%"
    rows = await db.fetch_all(
        """
        SELECT DISTINCT t.id
        FROM tracks t
        JOIN track_sources s ON s.track_id = t.id AND s.connector = 'local'
        WHERE t.title ILIKE :q
           OR t.artists_text ILIKE :q
           OR COALESCE(t.album, '') ILIKE :q
        ORDER BY t.id DESC
        LIMIT :l
        """,
        {"q": q, "l": limit},
    )
    return [r["id"] for r in rows]
