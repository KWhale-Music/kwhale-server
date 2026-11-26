"""DTO builders: canonical Track/Album/Artist dicts shared by all API responses.

TrackDTO shape (everywhere): {id, title, artists, album, duration_sec, year,
track_number, genre, isrc, cover_url, in_library, sources[{connector,
quality_tier}], album_id, artist_ids, is_favorite}. external_id is internal —
never leaves the API.

Batch builders take a list of ids and issue a handful of queries (rows,
sources, entity links, favorites) — no N+1. Entity ids are resolved by
normalized name (albums/artists are text-linked in tracks), which is fine for
a personal catalogue; a huge one would want functional indexes instead.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from .. import db
from .dedup import normalize_artist, normalize_title_or_raw
from .normalize import album_title_key, compact_artist_key, split_artists_strict

# TTL cache for the artist/album id indexes: build_track_dtos used to run a full
# SELECT over artists/albums on EVERY call (O(library) per request). These maps
# change rarely, so we rebuild at most once per _IDX_TTL seconds. _ensure_entity_rows
# still updates the cached map in-place when it materialises a missing entity.
_IDX_TTL = 60.0
_idx_cache: dict[str, tuple[float, Any]] = {}


def remember_artist(name: str, artist_id: int) -> None:
    """Keep the cached index fresh after an upsert creates an artist."""
    norm = normalize_artist(name)
    compact = compact_artist_key(name)
    if not norm and not compact:
        return
    cached = _idx_cache.get("artists")
    if cached is not None:
        if norm:
            cached[1].setdefault(norm, artist_id)
        if compact:
            cached[1].setdefault(compact, artist_id)


def remember_album(title: str, artist_name: str, album_id: int) -> None:
    """Keep the cached album index fresh after an upsert creates an album."""
    norm_title = normalize_title_or_raw(title)
    alb_title = album_title_key(title)
    norm_artist = normalize_artist(artist_name)
    comp_artist = compact_artist_key(artist_name)
    if not (norm_title or alb_title) or not (norm_artist or comp_artist):
        return
    cached = _idx_cache.get("albums")
    if cached is not None:
        for tk in dict.fromkeys(k for k in (norm_title, alb_title) if k):
            for ak in dict.fromkeys(k for k in (norm_artist, comp_artist) if k):
                cached[1].setdefault((tk, ak), album_id)


async def build_track_dtos(
    track_ids: list[int], user_id: int | None = None
) -> list[dict[str, Any]]:
    """TrackDTOs for the given ids, in the same order as the input."""
    if not track_ids:
        return []
    placeholders = [f":i{i}" for i in range(len(track_ids))]
    params = {f"i{i}": tid for i, tid in enumerate(track_ids)}
    rows = await db.fetch_all(
        f"SELECT * FROM tracks WHERE id IN ({', '.join(placeholders)})", params
    )
    src_rows = await db.fetch_all(
        f"""
        SELECT track_id, connector, quality_tier
        FROM track_sources WHERE track_id IN ({", ".join(placeholders)})
        ORDER BY quality_tier DESC
        """,
        params,
    )
    artist_index = await _artist_id_index()
    album_index = await _album_id_index()
    rows_by_id = {r["id"]: r for r in rows}
    favorite_ids = await _cluster_flags(track_ids, rows_by_id, user_id, "favorites")
    disliked_ids = await _cluster_flags(track_ids, rows_by_id, user_id, "dislikes")

    by_id = rows_by_id
    # Every track we serve must be navigable: create entity rows that are
    # missing from the graph (search-only tracks were never upserted).
    if by_id:
        await _ensure_entity_rows(list(by_id.values()), artist_index, album_index)

    sources_by_track: dict[int, list[dict[str, Any]]] = {}
    for s in src_rows:
        sources_by_track.setdefault(s["track_id"], []).append(
            {"connector": s["connector"], "quality_tier": s["quality_tier"]}
        )
    out: list[dict[str, Any]] = []
    for tid in track_ids:
        r = by_id.get(tid)
        if r is None:
            continue
        dto = _track_dto(r, sources_by_track.get(tid, []))
        artists = r["artists"] or []
        dto["album_id"] = _match_album_id(r["album"], artists, album_index)
        dto["artist_ids"] = [_lookup_artist_id(n, artist_index) for n in artists]
        dto["is_favorite"] = tid in favorite_ids
        dto["is_disliked"] = tid in disliked_ids
        out.append(dto)
    return out


def _lookup_artist_id(name: str, artist_index: dict[str, int]) -> int | None:
    direct = artist_index.get(normalize_artist(name)) or artist_index.get(compact_artist_key(name))
    if direct is not None:
        return direct
    for part in split_artists_strict(name):
        pid = artist_index.get(normalize_artist(part)) or artist_index.get(compact_artist_key(part))
        if pid is not None:
            return pid
    return None


async def track_dto(track_id: int, user_id: int | None = None) -> dict[str, Any] | None:
    dtos = await build_track_dtos([track_id], user_id)
    return dtos[0] if dtos else None


async def _artist_id_index() -> dict[str, int]:
    """normalized artist name -> artist id (first wins on duplicates)."""
    now = time.monotonic()
    cached = _idx_cache.get("artists")
    if cached is not None and now - cached[0] < _IDX_TTL:
        return cached[1]
    rows = await db.fetch_all("SELECT id, name FROM artists LIMIT 100000")
    index = await asyncio.to_thread(_build_artist_index, rows)
    _idx_cache["artists"] = (now, index)
    return index


def _build_artist_index(rows: list[dict[str, Any]]) -> dict[str, int]:
    """CPU part of the index build: normalize 56k names off the event loop."""
    index: dict[str, int] = {}
    for r in rows:
        key = normalize_artist(r["name"])
        if key and key not in index:
            index[key] = r["id"]
        ckey = compact_artist_key(r["name"])
        if ckey and ckey not in index:
            index[ckey] = r["id"]
    return index


def _expand_artist_entries(
    raw_names: list[str], artist_index: dict[str, int], single_names: set[str]
) -> list[str]:
    out: list[str] = []
    for name in raw_names:
        parts = split_artists_strict(name)
        if len(parts) <= 1:
            if name and name not in out:
                out.append(name)
            continue
        has_feat = any(m in name.lower() for m in ("feat.", "feat ", "ft.", "ft ", "featuring", "при уч."))
        known = sum(
            1
            for p in parts
            if normalize_artist(p) in artist_index
            or compact_artist_key(p) in artist_index
            or normalize_artist(p) in single_names
        )
        chosen = parts if (has_feat or known >= 2) else [name]
        for p in chosen:
            if p and p not in out:
                out.append(p)
    return out


async def _ensure_entity_rows(
    rows: list[dict[str, Any]],
    artist_index: dict[str, int],
    album_index: dict[tuple[str, str], int],
) -> None:
    """Materialise artist/album rows referenced by these tracks but missing.

    Navigation from the player (title -> album, artist -> artist card) needs
    real entity ids; tracks that only ever appeared in search results have
    none, so create source-less rows (the GC sweeps them when unused).
    """
    from .entity_upsert import ensure_album_by_title, ensure_artist_by_name

    # Collect solo names occurring across this batch so packed collabs like
    # "SALUKI, Toxi$" resolve cleanly against batch peers without shell rows.
    single_names: set[str] = set()
    for r in rows:
        for name in r["artists"] or []:
            if len(split_artists_strict(name)) <= 1:
                nk = normalize_artist(name)
                if nk:
                    single_names.add(nk)

    missing_artists: dict[str, str] = {}
    missing_albums: dict[tuple[str, str], tuple[Any, ...]] = {}
    for r in rows:
        resolved = _expand_artist_entries(list(r["artists"] or []), artist_index, single_names)
        for name in resolved:
            norm = normalize_artist(name)
            comp = compact_artist_key(name)
            if norm and norm not in artist_index and comp not in artist_index:
                missing_artists.setdefault(norm, name)
        title = r["album"]
        if title and resolved:
            artist_name = resolved[0]
            norm_t = normalize_title_or_raw(title)
            alb_t = album_title_key(title)
            norm_a = normalize_artist(artist_name)
            comp_a = compact_artist_key(artist_name)
            found = any(
                (tk, ak) in album_index
                for tk in (norm_t, alb_t)
                if tk
                for ak in (norm_a, comp_a)
                if ak
            )
            key = (norm_t, norm_a)
            if norm_t and norm_a and not found:
                missing_albums.setdefault(
                    key, (title, artist_name, r.get("year"), r.get("cover_url"))
                )

    for norm, name in missing_artists.items():
        artist_id = await ensure_artist_by_name(name)
        if artist_id:
            artist_index[norm] = artist_id
            comp = compact_artist_key(name)
            if comp:
                artist_index.setdefault(comp, artist_id)

    for (_norm_title, norm_artist), (title, artist_name, year, cover) in missing_albums.items():
        artist_id = artist_index.get(norm_artist) or artist_index.get(compact_artist_key(artist_name))
        if artist_id is None:
            continue
        album_id = await ensure_album_by_title(title, artist_id, year, cover)
        if album_id:
            remember_album(title, artist_name, album_id)


async def _album_id_index() -> dict[tuple[str, str], int]:
    """(normalized title, normalized artist name) -> album id."""
    now = time.monotonic()
    cached = _idx_cache.get("albums")
    if cached is not None and now - cached[0] < _IDX_TTL:
        return cached[1]
    rows = await db.fetch_all(
        """
        SELECT a.id, a.title, ar.name AS artist_name
        FROM albums a JOIN artists ar ON ar.id = a.artist_id
        """
    )
    index = await asyncio.to_thread(_build_album_index, rows)
    _idx_cache["albums"] = (now, index)
    return index


def _build_album_index(rows: list[dict[str, Any]]) -> dict[tuple[str, str], int]:
    index: dict[tuple[str, str], int] = {}
    for r in rows:
        t_raw = normalize_title_or_raw(r["title"])
        t_alb = album_title_key(r["title"])
        a_norm = normalize_artist(r["artist_name"] or "")
        a_comp = compact_artist_key(r["artist_name"] or "")
        for tk in dict.fromkeys(k for k in (t_raw, t_alb) if k):
            for ak in dict.fromkeys(k for k in (a_norm, a_comp) if k):
                if (tk, ak) not in index:
                    index[(tk, ak)] = r["id"]
    return index


def _match_album_id(
    album_title: str | None,
    artists: list[str],
    album_index: dict[tuple[str, str], int],
) -> int | None:
    if not album_title:
        return None
    title_keys = [k for k in (normalize_title_or_raw(album_title), album_title_key(album_title)) if k]
    if not title_keys:
        return None
    candidates: list[str] = []
    for artist in artists:
        if artist and artist not in candidates:
            candidates.append(artist)
    for artist in artists:
        for part in split_artists_strict(artist):
            if part and part not in candidates:
                candidates.append(part)
    for tk in title_keys:
        for artist in candidates:
            for ak in (normalize_artist(artist), compact_artist_key(artist)):
                if ak:
                    album_id = album_index.get((tk, ak))
                    if album_id:
                        return album_id
    return None


async def _cluster_flags(
    track_ids: list[int], rows_by_id: dict[int, dict[str, Any]], user_id: int | None, table: str
) -> set[int]:
    """Requested ids flagged via ANY row of their identity cluster.

    A like lives on one tracks row, but the same recording may exist as a
    twin row (another source, ±15 s duration). Flag state must follow the
    recording, not the row — «лайкнутый трек отображается нелайкнутым».
    Candidates are narrowed by the identity_hash index in SQL, then the ±15 s
    window is applied in Python (identity.py owns the rule).
    """
    from .identity import duration_close, identity_hash

    if user_id is None or not track_ids:
        return set()
    placeholders = [f":f{i}" for i in range(len(track_ids))]
    params: dict[str, Any] = {"u": user_id}
    params.update({f"f{i}": tid for i, tid in enumerate(track_ids)})

    flagged: set[int] = set()
    # Direct flags on the requested rows themselves (covers legacy NULL-hash rows).
    direct = await db.fetch_all(
        f"SELECT track_id FROM {table} "
        f"WHERE user_id = :u AND track_id IN ({', '.join(placeholders)})",
        params,
    )
    flagged.update(int(r["track_id"]) for r in direct)

    # Cluster flags: any favorited/disliked twin row counts.
    hashes = {identity_hash(r.get("title"), r.get("artists")) for r in rows_by_id.values()}
    hash_list = sorted(hashes)
    h_placeholders = [f":h{i}" for i in range(len(hash_list))]
    h_params: dict[str, Any] = {"u": user_id}
    h_params.update({f"h{i}": h for i, h in enumerate(hash_list)})
    cands = await db.fetch_all(
        f"""
        SELECT DISTINCT f.track_id, x.title, x.artists, x.duration_sec, x.identity_hash
        FROM {table} f JOIN tracks x ON x.id = f.track_id
        WHERE f.user_id = :u AND x.identity_hash IN ({", ".join(h_placeholders)})
        """,
        h_params,
    )
    by_hash: dict[str, list[dict[str, Any]]] = {}
    for c in cands:
        by_hash.setdefault(c["identity_hash"], []).append(c)
    for tid, r in rows_by_id.items():
        h = identity_hash(r.get("title"), r.get("artists"))
        for c in by_hash.get(h, ()):
            if duration_close(r.get("duration_sec"), c.get("duration_sec")):
                flagged.add(tid)
                break
    return flagged


def _track_dto(row: dict[str, Any], sources: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "id": row["id"],
        "title": row["title"],
        "artists": row["artists"],
        "album": row["album"],
        "duration_sec": row["duration_sec"],
        "year": row.get("year"),
        "track_number": row.get("track_number"),
        "track_total": row.get("track_total"),
        "disc_number": row.get("disc_number"),
        "disc_total": row.get("disc_total"),
        "explicit": row.get("explicit"),
        "has_lyrics": bool(row.get("has_lyrics")),
        "genre": row.get("genre"),
        "isrc": row["isrc"],
        "cover_url": row["cover_url"],
        "in_library": any(s["connector"] == "local" for s in sources),
        "sources": sources,
    }


def page(items: list[dict[str, Any]], total: int, limit: int, offset: int) -> dict[str, Any]:
    return {"items": items, "total": total, "limit": limit, "offset": offset}
