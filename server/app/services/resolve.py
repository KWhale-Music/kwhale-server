"""Online entity resolution by name (cards opened from liked tracks).

When a card is opened for an entity that has no connector sources yet (rows
materialised from a liked track, or a stale row), search the connectors by
name, keep the best normalized match and let the caller link it as a source.
Results (including negatives) are cached for RESOLVE_TTL_HOURS so repeated
opens do not hammer the network -- connectors additionally cache their own
album/artist payloads for 24h.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from .. import db
from ..connectors import get_connectors
from ..connectors.base import AlbumMeta, ArtistMeta
from .dedup import normalize_artist, normalize_title_or_raw
from .search_service import cache_get, cache_put

log = logging.getLogger("kwhale.resolve")

RESOLVE_TTL_HOURS = 6
PREFERRED_ORDER = ("icm", "yandex", "soundcloud", "ytmusic")


def _connector_rank(name: str) -> int:
    try:
        return PREFERRED_ORDER.index(name)
    except ValueError:
        return len(PREFERRED_ORDER)


async def _search_one(conn: Any, kind: str, query: str, limit: int) -> list[Any]:
    if not await conn.is_available():
        return []
    if kind == "album":
        return await conn.search_albums(query, limit) or []
    return await conn.search_artists(query, limit) or []


async def _fan_out(kind: str, query: str, limit: int = 8) -> list[Any]:
    connectors = get_connectors()
    ordered = sorted(connectors.items(), key=lambda kv: _connector_rank(kv[0]))
    results = await asyncio.gather(
        *(_search_one(conn, kind, query, limit) for _, conn in ordered),
        return_exceptions=True,
    )
    metas: list[Any] = []
    for (name, _), result in zip(ordered, results, strict=False):
        if isinstance(result, BaseException):
            log.warning("resolve %s via %s failed: %s", kind, name, result)
            continue
        metas.extend(result)
    return metas


def _artist_cache_entry(meta: ArtistMeta | None) -> dict[str, Any]:
    if meta is None:
        return {"found": False}
    return {
        "found": True,
        "meta": {
            "connector": meta.connector,
            "external_id": meta.external_id,
            "name": meta.name,
            "cover_url": meta.cover_url,
            "genre": meta.genre,
        },
    }


def _album_cache_entry(meta: AlbumMeta | None) -> dict[str, Any]:
    if meta is None:
        return {"found": False}
    return {
        "found": True,
        "meta": {
            "connector": meta.connector,
            "external_id": meta.external_id,
            "title": meta.title,
            "artist_name": meta.artist_name,
            "year": meta.year,
            "cover_url": meta.cover_url,
            "track_count": meta.track_count,
            "genre": meta.genre,
        },
    }


async def find_artist_online(name: str) -> ArtistMeta | None:
    """Best connector artist matching [name] exactly (normalized), or None."""
    norm = normalize_artist(name)
    if not norm:
        return None
    cache_key = f"resolve:artist:{norm}"
    cached = await cache_get(cache_key)
    if isinstance(cached, dict):
        if not cached.get("found"):
            return None
        meta = cached.get("meta") or {}
        if meta.get("connector") and meta.get("external_id"):
            return ArtistMeta(**meta, raw={})
        return None

    metas = await _fan_out("artist", name)
    match = next((m for m in metas if normalize_artist(m.name) == norm), None)
    await cache_put(cache_key, "resolve", _artist_cache_entry(match), ttl_hours=RESOLVE_TTL_HOURS)
    return match


async def find_album_online(title: str, artist_name: str) -> AlbumMeta | None:
    """Best connector album matching title+artist (normalized), or None."""
    norm_title = normalize_title_or_raw(title)
    norm_artist = normalize_artist(artist_name)
    if not norm_title:
        return None
    cache_key = f"resolve:album:{norm_title}:{norm_artist}"
    cached = await cache_get(cache_key)
    if isinstance(cached, dict):
        if not cached.get("found"):
            return None
        meta = cached.get("meta") or {}
        if meta.get("connector") and meta.get("external_id"):
            return AlbumMeta(**meta, raw={})
        return None

    query = f"{artist_name} {title}".strip()
    metas = await _fan_out("album", query)
    match = next(
        (
            m
            for m in metas
            if normalize_title_or_raw(m.title) == norm_title
            and (not norm_artist or normalize_artist(m.artist_name or "") == norm_artist)
        ),
        None,
    )
    await cache_put(cache_key, "resolve", _album_cache_entry(match), ttl_hours=RESOLVE_TTL_HOURS)
    return match


async def link_album_via_artist(
    album_id: int,
    title: str,
    artist_id: int,
    artist_name: str,
) -> bool:
    """Fallback resolution: match the album inside the artist's discography.

    Connector album search misses titles that are symbols or that providers
    label differently ('†' by Justice); the artist's album list has them.
    """
    from ..connectors import get_connector
    from .entity_upsert import link_album_source, upsert_artist

    norm_title = normalize_title_or_raw(title)
    if not norm_title:
        return False

    meta_artist = await find_artist_online(artist_name)
    if meta_artist is not None:
        await upsert_artist(meta_artist)  # links the artist source

    src = await db.fetch_one(
        """
        SELECT connector, external_id FROM artist_sources
        WHERE artist_id = :i AND available
        ORDER BY connector = 'icm' DESC
        LIMIT 1
        """,
        {"i": artist_id},
    )
    if not src:
        return False
    connector = get_connector(src["connector"])
    if connector is None:
        return False
    try:
        metas = await connector.artist_albums(src["external_id"], 100) or []
    except Exception as e:
        log.warning("artist_albums failed for %s: %s", artist_name, e)
        return False

    match = next(
        (m for m in metas if normalize_title_or_raw(m.title) == norm_title),
        None,
    )
    if match is None:
        return False
    await link_album_source(album_id, match)
    return True
