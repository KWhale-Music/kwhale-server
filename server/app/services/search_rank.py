"""Search relevance: query match first, popularity second.

The old ordering was source-first (ICM ahead of everything, then quality tier)
and round-robin across connectors. That is good for catalogue coverage but bad
for "what did the user mean": searching `Nirvana` could put a track *named*
"Nirvana" by another artist ahead of the band, and the client's "best result"
simply took the first item of a section.

This module is intentionally dependency-free so the same rules can be unit
tested and reused by the search router. Popularity is a *tie breaker*: exact
and prefix matches always win over a fuzzy but popular row.
"""

from __future__ import annotations

import math
from difflib import SequenceMatcher
from typing import Any

from .dedup import duration_close, track_identity_key
from .normalize import (
    album_title_key,
    artists_compatible,
    compact_artist_key,
    dedup_artist_key,
    dedup_title_key,
    split_uploader_title,
)


def _field_key(text: str | None) -> str:
    return dedup_title_key(text or "")


def _artist_key(text: str | None) -> str:
    return dedup_artist_key(text or "")


def match_score(query: str, *fields: str | None) -> float:
    """0..1 relevance of a query against one or more entity fields.

    Exact > prefix > all query words present > one word present > substring >
    fuzzy ratio. Taking the best field lets a track match by its artist even
    when the title is unrelated.
    """
    q = _field_key(query)
    if not q:
        return 0.0
    q_tokens = q.split()
    best = 0.0
    matched_fields = 0
    for field in fields:
        f = _field_key(field)
        if not f:
            continue
        if f == q:
            score = 1.0
        elif f.startswith(q) or q.startswith(f):
            score = 0.88
        else:
            f_tokens = set(f.split())
            if q_tokens and all(token in f_tokens for token in q_tokens):
                score = 0.78
            elif any(token in f_tokens for token in q_tokens):
                score = 0.62
            elif q in f:
                score = 0.55
            else:
                score = 0.5 * SequenceMatcher(None, q, f).ratio()
        if score >= 0.78:
            matched_fields += 1
        best = max(best, score)
    # A track whose title AND artist both match the query is a little stronger
    # than one that matches only one field.
    if matched_fields >= 2:
        best = min(1.0, best + 0.05)
    return best


def _artist_exact(query: str, artists: list[str]) -> bool:
    q = _artist_key(query)
    return bool(q) and any(_artist_key(a) == q for a in artists if a)


def _artist_names(item: Any) -> list[str]:
    """Read artist names from a pydantic DTO or a snake_case dict."""
    if isinstance(item, dict):
        artists = item.get("artists") or []
        names = []
        for artist in artists:
            if isinstance(artist, dict):
                name = artist.get("name")
            else:
                name = getattr(artist, "name", None)
            if name:
                names.append(name)
        if not names and item.get("artist"):
            artist = item["artist"]
            names.append(artist.get("name") if isinstance(artist, dict) else str(artist))
        return names
    artists = getattr(item, "artists", None) or []
    names = [getattr(a, "name", None) for a in artists]
    return [n for n in names if n]


def _value(item: Any, key: str, default: Any = None) -> Any:
    if isinstance(item, dict):
        return item.get(key, default)
    return getattr(item, key, default)


def _id(item: Any) -> str:
    return str(_value(item, "id", "") or "")


def _title(item: Any) -> str:
    return _value(item, "title") or _value(item, "name") or ""


def _log_norm(value: float, maximum: float) -> float:
    if maximum <= 0:
        return 0.0
    return math.log1p(max(0.0, value)) / math.log1p(maximum)


def track_score(query: str, item: Any) -> float:
    title = _title(item)
    artists = _artist_names(item)
    score = match_score(query, title, *artists)
    if _artist_exact(query, artists):
        score += 0.15
    if _value(item, "is_favorite") or _value(item, "isFavorited"):
        score += 0.03
    if _value(item, "is_downloaded"):
        score += 0.02
    return score


def album_score(query: str, item: Any) -> float:
    title = _title(item)
    artists = _artist_names(item)
    score = match_score(query, title, *artists)
    if _field_key(query) and _field_key(query) == _field_key(title):
        # Exact album title is a strong signal: "Nevermind" is an album even
        # when an obscure artist with that name exists.
        score += 0.1
    if _artist_exact(query, artists):
        score += 0.12
    if _value(item, "is_favorite") or _value(item, "isFavorite"):
        score += 0.03
    count = int(_value(item, "track_count", _value(item, "trackCount", 0)) or 0)
    return score + 0.03 * _log_norm(count, 50)


def artist_score(query: str, item: Any) -> float:
    name = _value(item, "name") or ""
    score = match_score(query, name)
    if _artist_key(query) == _artist_key(name):
        score += 0.2
    count = int(_value(item, "album_count", _value(item, "albumCount", 0)) or 0)
    if _value(item, "is_favorite") or _value(item, "isFavorite"):
        score += 0.03
    return score + 0.05 * _log_norm(count, 30)


def playlist_score(query: str, item: Any) -> float:
    score = match_score(query, _title(item))
    count = int(_value(item, "track_count", _value(item, "trackCount", 0)) or 0)
    return score + 0.02 * _log_norm(count, 200)


def _recency(item: Any) -> tuple[int, int]:
    """Newest-first tie breaker: release year, then higher id (newer ingest)."""
    try:
        year = int(_value(item, "year", 0) or 0)
    except (TypeError, ValueError):
        year = 0
    try:
        ident = int(_id(item))
    except ValueError:
        ident = 0
    return (year, ident)


def sort_tracks(query: str, items: list[Any]) -> list[Any]:
    return sorted(
        items,
        key=lambda item: (-track_score(query, item),) + tuple(-v for v in _recency(item)) + (_title(item).lower(),),
    )


def sort_albums(query: str, items: list[Any]) -> list[Any]:
    return sorted(items, key=lambda item: (-album_score(query, item), _title(item).lower()))


def sort_artists(query: str, items: list[Any]) -> list[Any]:
    return sorted(items, key=lambda item: (-artist_score(query, item), _title(item).lower()))


def sort_playlists(query: str, items: list[Any]) -> list[Any]:
    return sorted(items, key=lambda item: (-playlist_score(query, item), _title(item).lower()))


def pick_top(
    query: str,
    tracks: list[Any],
    albums: list[Any],
    artists: list[Any],
    playlists: list[Any],
    popularity: dict[str, float] | None = None,
) -> dict[str, Any] | None:
    """Return the single best result as a typed envelope.

    Type priors encode what people usually mean by a short query: an exact
    artist name is the artist, an exact album title is the album; otherwise a
    track wins. Popularity only breaks ties within a type.
    """
    pop = popularity or {}

    def _pop(kind: str, item: Any) -> float:
        return pop.get(f"{kind}:{_id(item)}", 0.0)

    candidates: list[tuple[float, int, str, Any]] = []
    for item in artists[:5]:
        candidates.append((artist_score(query, item) + 0.15 * _pop("artist", item), 4, "artist", item))
    for item in albums[:5]:
        candidates.append((album_score(query, item) + 0.15 * _pop("album", item), 3, "album", item))
    for item in tracks[:8]:
        candidates.append((track_score(query, item) + 0.1 * _pop("track", item), 2, "track", item))
    for item in playlists[:5]:
        candidates.append((playlist_score(query, item), 1, "playlist", item))
    if not candidates:
        return None
    score, _priority, kind, item = max(
        candidates, key=lambda row: (row[0], row[1]) + _recency(row[3])
    )
    if score < 0.55:
        return None
    return {"type": kind, kind: item}


# ── Response-level dedup ─────────────────────────────────────────────────────
# One search fan-out can surface the same artist/album row twice (remote
# upserts re-enter through several groups) or several rows with the same
# display name (historical duplicates: «Holidayboy» x3). The client shows these
# as "множество дублирующихся исполнителей", so collapse them before ranking:
# same id or same normalized name -> keep the richest entry (has cover, more
# albums, favorited).


def _artist_name_key(item: Any) -> str:
    name = _value(item, "name") or ""
    return compact_artist_key(name) or _artist_key(name)


def _album_name_key(item: Any) -> str:
    raw_title = _title(item)
    artists = _artist_names(item)
    split_dash = split_uploader_title(raw_title)
    if split_dash and artists:
        left, right = split_dash
        if artists_compatible([left], artists, allow_subset=True):
            raw_title = right
    tkey = album_title_key(raw_title)
    akeys = sorted({compact_artist_key(a) or _artist_key(a) for a in artists if a})
    return f"{','.join(akeys)}|{tkey}"


def _duration_sec(item: Any) -> int:
    sec = _value(item, "duration_sec")
    if sec is None:
        ms = _value(item, "duration_ms") or _value(item, "durationMs")
        if ms is not None:
            try:
                sec = int(ms) // 1000
            except (TypeError, ValueError):
                sec = 0
    try:
        return int(sec or 0)
    except (TypeError, ValueError):
        return 0


def _track_richness(item: Any) -> tuple[int, int, int, int, int]:
    in_lib = 1 if (_value(item, "in_library") or _value(item, "is_downloaded")) else 0
    fav = 1 if (_value(item, "is_favorite") or _value(item, "isFavorited")) else 0
    srcs = _value(item, "sources") or []
    cover = _value(item, "cover") or _value(item, "cover_url")
    has_cover = 0
    if isinstance(cover, dict):
        has_cover = 1 if cover.get("id") not in (None, "", "placeholder") else 0
    elif isinstance(cover, str):
        has_cover = 1 if cover.strip() else 0
    elif cover is not None:
        has_cover = 0 if getattr(cover, "id", None) in (None, "", "placeholder") else 1
    dur = _duration_sec(item)
    return (in_lib, fav, len(srcs), has_cover, 1 if dur > 0 else 0)


def dedupe_tracks(items: list[Any]) -> list[Any]:
    """Collapse duplicate TrackDTOs in a search/catalog response before ranking.

    Prevents minor upstream drift (e.g. "Artist — Title" SoundCloud reuploads,
    feat-in-title vs packed artists, missing duration=0 vs full track) from
    rendering the exact same recording twice in search results regardless of order.
    """
    out: list[Any] = []
    seen_ids: set[str] = set()
    by_title: dict[str, list[int]] = {}
    # Order-independent lookup for uploader prefixes whose channel didn't match artist
    by_split_right: dict[str, list[tuple[int, str]]] = {}

    for item in items:
        item_id = _id(item)
        if item_id and item_id in seen_ids:
            continue
        if item_id:
            seen_ids.add(item_id)

        title = _title(item)
        artists = _artist_names(item)
        dur = _duration_sec(item)
        tkey, t_parts, _ = track_identity_key(title, artists, dur)
        split_dash = split_uploader_title(title)
        cand_participants = [
            p for p in artists
        ]
        # Build alternate (inf_title, inf_artists) when title looks like "Artist — Title"
        alt_pairs: list[tuple[str, list[str]]] = []
        if split_dash:
            left, right = split_dash
            rtkey, rparts, _ = track_identity_key(right, [left], dur)
            if rtkey:
                alt_pairs.append((rtkey, list(rparts) or [left]))
        if len(artists) == 1 and any(sep in title for sep in (",", "&")):
            stkey, sparts, _ = track_identity_key(artists[0], [title], dur)
            if stkey:
                alt_pairs.append((stkey, list(sparts) or [title]))

        matched_pos: int | None = None
        for idx in by_title.get(tkey, []):
            prev = out[idx]
            if duration_close(dur, _duration_sec(prev)) and artists_compatible(
                artists or list(t_parts), _artist_names(prev), allow_subset=True
            ):
                matched_pos = idx
                break
        if matched_pos is None:
            for alt_tkey, alt_artists in alt_pairs:
                for idx in by_title.get(alt_tkey, []):
                    prev = out[idx]
                    if duration_close(dur, _duration_sec(prev)) and artists_compatible(
                        alt_artists, _artist_names(prev), allow_subset=True
                    ):
                        matched_pos = idx
                        break
                if matched_pos is not None:
                    break
        if matched_pos is None and tkey in by_split_right:
            for idx, prev_left in by_split_right[tkey]:
                prev = out[idx]
                if duration_close(dur, _duration_sec(prev)) and artists_compatible(
                    artists or list(t_parts), [prev_left], allow_subset=True
                ):
                    matched_pos = idx
                    break

        if matched_pos is not None:
            if _track_richness(item) > _track_richness(out[matched_pos]):
                out[matched_pos] = item
                # Update keys under the richer canonical form
                pos_key, _, _ = track_identity_key(_title(item), _artist_names(item), dur)
                if pos_key and matched_pos not in by_title.get(pos_key, []):
                    by_title.setdefault(pos_key, []).append(matched_pos)
            continue

        pos = len(out)
        out.append(item)
        if tkey:
            by_title.setdefault(tkey, []).append(pos)
        for alt_tkey, alt_artists in alt_pairs:
            if alt_tkey and alt_artists:
                by_split_right.setdefault(alt_tkey, []).append((pos, ", ".join(alt_artists)))
        _ = cand_participants
    return out


def _artist_richness(item: Any) -> tuple[int, int, int]:
    avatar = _value(item, "avatar")
    has_cover = 0
    if isinstance(avatar, dict):
        has_cover = 1 if avatar.get("id") not in (None, "", "placeholder") else 0
    elif avatar is not None:
        has_cover = 0 if getattr(avatar, "id", None) in (None, "", "placeholder") else 1
    count = int(_value(item, "album_count", _value(item, "albumCount", 0)) or 0)
    return (has_cover, count, 1 if _value(item, "is_favorite") else 0)


def _album_richness(item: Any) -> tuple[int, int, int]:
    cover = _value(item, "cover")
    has_cover = 0
    if isinstance(cover, dict):
        has_cover = 1 if cover.get("id") not in (None, "", "placeholder") else 0
    elif cover is not None:
        has_cover = 0 if getattr(cover, "id", None) in (None, "", "placeholder") else 1
    count = int(_value(item, "track_count", _value(item, "trackCount", 0)) or 0)
    return (has_cover, count, 1 if _value(item, "is_favorite") else 0)


def dedupe_artists(items: list[Any]) -> list[Any]:
    """Collapse duplicate artist entries (same id or same normalized name)."""
    return _dedupe(items, lambda a: _id(a), _artist_name_key, _artist_richness)


def dedupe_albums(items: list[Any]) -> list[Any]:
    """Collapse duplicate album entries (same id or same normalized title+artist)."""
    return _dedupe(items, lambda a: _id(a), _album_name_key, _album_richness)


def _dedupe(items: list[Any], id_key, name_key, richness) -> list[Any]:
    out: list[Any] = []
    index: dict[str, int] = {}  # name key -> position in out
    seen_ids: set[str] = set()
    for item in items:
        item_id = id_key(item)
        if item_id and item_id in seen_ids:
            continue
        if item_id:
            seen_ids.add(item_id)
        key = name_key(item)
        if not key:
            out.append(item)
            continue
        pos = index.get(key)
        if pos is None:
            index[key] = len(out)
            out.append(item)
        elif richness(item) > richness(out[pos]):
            out[pos] = item
    return out
