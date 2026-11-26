"""Dedup: merge connector results into canonical tracks, albums, and artists.

Strategy:
  1. ISRC exact match when both sides have it (strongest recording signal,
     guarded against 30-second preview snippets colliding with full songs)
  2. Multi-signal match: normalized title (feat/prod/upload noise stripped,
     Remix/Live/Slowed/Acoustic/Demo/Instrumental kept distinct) + compatible
     participants (handles "A & B" vs ["A", "B"] vs "(feat. B)", reordered
     credits, spacing drift, homoglyphs, mixed-script, and RU<->LAT translit)
     + proportional duration check across all known non-zero durations
  3. Pass-2 uploader / swapped cleanup: folds SoundCloud/YT channel reuploads
     ("Artist — Track" / swapped title<->artist) when they match a canonical
     Pass-1 group
Winner per group = ICM (if present) or highest quality_tier; all sources kept.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from ..connectors.base import TrackMeta
from .normalize import (
    album_title_key,
    artist_identity_keys,
    artists_compatible,
    compact_artist_key,
    dedup_artist_key,
    dedup_title_key,
    extract_featured_artists,
    flatten_artist_names,
    split_artists,
    split_uploader_title,
    translit_ru_to_lat,
)

_PREVIEW_MAX_SEC = 35
_FULL_TRACK_MIN_SEC = 65


def normalize_title(text: str) -> str:
    return dedup_title_key(text)


def normalize_artist(text: str) -> str:
    return dedup_artist_key(text)


def normalize_title_or_raw(text: str) -> str:
    """normalize_title with a raw fallback for symbol-only titles (e.g. '†').

    normalize_title strips non-word characters, so '†' collapses to an empty
    string; entity linking/resolution would then treat it as "no title".
    """
    return normalize_title(text) or text.strip().lower()


def track_identity_key(
    title: str | None, artists: list[str] | None, duration_sec: int | None
) -> tuple[str, tuple[str, ...], int]:
    """Presentation-level identity of a track: normalized title + artists + duration.

    The DB can hold several rows for the same recording (historical upserts
    from providers with missing durations); lists should show one.
    """
    raw_title = title or ""
    clean_title, featured = extract_featured_artists(raw_title)
    all_artists = [*(artists or []), *featured]
    flat = [p for a in all_artists for p in (split_artists(a) or [a]) if p]
    norm_set = {normalize_artist(a) for a in flat if normalize_artist(a)}
    return (
        normalize_title_or_raw(clean_title),
        tuple(sorted(norm_set)),
        duration_sec or 0,
    )


def presentation_key(
    title: str | None, artists: list[str] | None
) -> tuple[str, tuple[str, ...]]:
    """Duration-free half of track_identity_key (same recording, noisy lengths)."""
    return track_identity_key(title, artists, 0)[:2]


def duration_close(a: int, b: int, tolerance: int | None = None) -> bool:
    """Check whether two durations can represent the same recording.

    Missing duration (0/None) never disproves a single comparison, while preview
    snippets (<=35s) are barred from merging with full-length recordings (>=65s).
    When `tolerance` is omitted, uses a proportional window capped at ±15s
    (matching identity.py / merge.py for normal songs, tighter for interludes).
    """
    if not a or not b:
        return True
    if (a <= _PREVIEW_MAX_SEC and b >= _FULL_TRACK_MIN_SEC) or (
        b <= _PREVIEW_MAX_SEC and a >= _FULL_TRACK_MIN_SEC
    ):
        return False
    if tolerance is None:
        tol = min(15, max(4, round(min(a, b) * 0.12)))
        return abs(a - b) <= tol
    return abs(a - b) <= tolerance


def _track_participants(title: str | None, artists: list[str] | None) -> tuple[str, list[str]]:
    """Extract clean title and merged participant list (unfolding self-prefix & feats)."""
    raw_title = title or ""
    raw_artists = [a for a in (artists or []) if a and a.strip()]
    split_dash = split_uploader_title(raw_title)
    if split_dash and raw_artists:
        left, right = split_dash
        if artists_compatible(flatten_artist_names([left]), raw_artists, allow_subset=True):
            raw_title = right
    stripped_title, featured = extract_featured_artists(raw_title)
    participants = flatten_artist_names(raw_artists)
    existing_keys = [artist_identity_keys(p) for p in participants]
    for f in featured:
        fkeys = artist_identity_keys(f)
        if not any(fkeys & ek for ek in existing_keys):
            participants.append(f)
            existing_keys.append(fkeys)
    return normalize_title_or_raw(stripped_title), participants


def _title_lookup_keys(norm_title: str) -> list[str]:
    """Exact normalized title key plus transliterated bucket for RU<->LAT."""
    if not norm_title:
        return []
    keys = [norm_title]
    tr = translit_ru_to_lat(norm_title)
    if tr and tr != norm_title and len(tr.replace(" ", "")) >= 4:
        keys.append(f"tr:{tr}")
    return keys


def collapse_track_rows(rows: list[dict], tolerance: int = 15) -> list[dict]:
    """Collapse rows for the same recording, keeping the first of each cluster.

    Rows arrive pre-sorted (local/favorite/richest first), so the best copy wins.
    Uses clean participant compatibility (handling "(feat. X)", "A & B",
    "Fortuna812,madkid" vs ["FORTUNA 812", "madk1d"], homoglyphs, translit)
    and uploader-prefix resolution alongside ±tolerance seconds duration match.
    """
    by_title: dict[str, list[tuple[list[str], list[int]]]] = defaultdict(list)
    out: list[dict] = []
    deferred: list[dict] = []

    def _try_place(
        row: dict, t_key: str, participants: list[str], dur: int, *, register: bool
    ) -> bool:
        for bkey in _title_lookup_keys(t_key):
            for known_parts, known_durs in by_title.get(bkey, ()):
                if not artists_compatible(known_parts, participants):
                    continue
                if dur and known_durs and not all(duration_close(d, dur, tolerance) for d in known_durs):
                    continue
                if dur and dur not in known_durs:
                    known_durs.append(dur)
                return True
        if register:
            durs = [dur] if dur else []
            for bkey in _title_lookup_keys(t_key):
                by_title[bkey].append((participants, durs))
        return False

    for row in rows:
        t_key, participants = _track_participants(row.get("title"), row.get("artists"))
        if not t_key:
            out.append(row)
            continue
        dur = int(row.get("duration_sec") or 0)
        if _try_place(row, t_key, participants, dur, register=False):
            continue
        # Defer potential SoundCloud uploader/swapped rows for Pass-2 resolution
        raw_title = row.get("title") or ""
        split_dash = split_uploader_title(raw_title)
        if split_dash:
            left, right = split_dash
            if not artists_compatible(flatten_artist_names([left]), row.get("artists")):
                deferred.append(row)
                continue
        _try_place(row, t_key, participants, dur, register=True)
        out.append(row)

    for row in deferred:
        raw_title = row.get("title") or ""
        dur = int(row.get("duration_sec") or 0)
        split_dash = split_uploader_title(raw_title)
        matched = False
        if split_dash:
            left, right = split_dash
            inf_title, inf_parts = _track_participants(right, [left])
            if inf_title and inf_parts:
                matched = _try_place(row, inf_title, inf_parts, dur, register=False)
        if not matched:
            t_key, participants = _track_participants(raw_title, row.get("artists"))
            if not _try_place(row, t_key, participants, dur, register=True):
                out.append(row)
    return out


def dedup_key(meta: TrackMeta) -> str:
    """Canonical track identity key string 'primary_artist|normalized_title'."""
    t_key, participants = _track_participants(meta.title, meta.artists)
    primary = normalize_artist(participants[0]) if participants else ""
    return f"{primary}|{t_key}"


def round_robin(groups: list[dict], limit: int) -> list[dict]:
    """Interleave dedup groups across winner connectors before truncation.

    Without this, groups are cut in first-seen order and connectors that fan
    out last silently disappear from small-limit searches (Phase-2 journal).
    """
    buckets: dict[str, list[dict]] = {}
    for g in groups:
        buckets.setdefault(getattr(g["winner"], "connector", ""), []).append(g)
    out: list[dict] = []
    while len(out) < limit and any(buckets.values()):
        for name in list(buckets):
            bucket = buckets[name]
            if bucket:
                out.append(bucket.pop(0))
                if len(out) >= limit:
                    break
    return out


class _Group:
    def __init__(self, primary: Any, title_key: str = "", participants: list[str] | None = None):
        self.winner: Any = primary
        self.members: list[Any] = [primary]
        self.title_key: str = title_key
        self.participants: list[str] = list(participants or [])
        dur = int(getattr(primary, "duration_sec", 0) or 0)
        self.durations: list[int] = [dur] if dur > 0 else []
        isrc = getattr(primary, "isrc", None)
        self.isrcs: set[str] = {isrc} if isrc else set()

    def duration_compatible(self, dur: int) -> bool:
        if not dur or not self.durations:
            return True
        return all(duration_close(d, dur) for d in self.durations)

    def add(self, meta: Any, participants: list[str] | None = None) -> None:
        self.members.append(meta)
        dur = int(getattr(meta, "duration_sec", 0) or 0)
        if dur > 0 and dur not in self.durations:
            self.durations.append(dur)
        isrc = getattr(meta, "isrc", None)
        if isrc:
            self.isrcs.add(isrc)
        for p in participants or []:
            if p not in self.participants:
                self.participants.append(p)

        tier = getattr(meta, "quality_tier", None)
        winner_tier = getattr(self.winner, "quality_tier", None)
        meta_exp = bool(getattr(meta, "explicit", False))
        win_exp = bool(getattr(self.winner, "explicit", False))
        any_exp = meta_exp or win_exp
        if getattr(meta, "connector", "") == "icm" and getattr(self.winner, "connector", "") != "icm":
            self.winner = meta
        elif getattr(self.winner, "connector", "") != "icm":
            if tier is not None and winner_tier is not None:
                if (tier, meta_exp) > (winner_tier, win_exp):
                    self.winner = meta
        if any_exp and hasattr(self.winner, "explicit"):
            self.winner.explicit = True


def dedup_tracks(metas: list[TrackMeta]) -> list[dict]:
    """Group metas into canonical tracks.

    Returns list of {'winner': TrackMeta, 'sources': [TrackMeta...]}
    in stable order (first-seen group order).
    """
    by_isrc: dict[str, _Group] = {}
    by_title: dict[str, list[_Group]] = defaultdict(list)
    ordered_groups: list[_Group] = []
    deferred: list[TrackMeta] = []

    def _register_group(g: _Group) -> None:
        ordered_groups.append(g)
        for isrc in g.isrcs:
            by_isrc.setdefault(isrc, g)
        for bkey in _title_lookup_keys(g.title_key):
            by_title[bkey].append(g)

    def _find_match(
        t_key: str,
        participants: list[str],
        dur: int,
        isrc: str | None,
        *,
        is_reupload: bool = False,
    ) -> _Group | None:
        for bkey in _title_lookup_keys(t_key):
            for g in by_title.get(bkey, ()):
                if isrc and g.isrcs and isrc not in g.isrcs:
                    continue
                if not artists_compatible(g.participants, participants):
                    continue
                if not g.duration_compatible(dur):
                    continue
                # Guard uploader reuploads against stealing explicit remixes/editions
                if is_reupload and bkey != g.title_key and not (
                    translit_ru_to_lat(t_key) == translit_ru_to_lat(g.title_key)
                ):
                    continue
                return g
        return None

    for m in metas:
        t_key, participants = _track_participants(m.title, m.artists)
        if not t_key and not m.isrc:
            continue
        dur = int(m.duration_sec or 0)

        if m.isrc:
            group = by_isrc.get(m.isrc)
            if group and group.duration_compatible(dur):
                group.add(m, participants)
                continue

        # Check if candidate looks like an uploader/channel reupload ("Artist — Title")
        # or swapped "(artist=[Title], title='Artist, CoArtist')" for Pass 2.
        split_dash = split_uploader_title(m.title or "")
        if not m.isrc and split_dash and m.artists:
            left, _right = split_dash
            if not artists_compatible(flatten_artist_names([left]), m.artists, allow_subset=True):
                deferred.append(m)
                continue
        if (
            not m.isrc
            and len(m.artists) == 1
            and any(sep in (m.title or "") for sep in (",", "&"))
            and "," not in m.artists[0]
            and "&" not in m.artists[0]
        ):
            deferred.append(m)
            continue

        matched = _find_match(t_key, participants, dur, m.isrc) if t_key else None
        if matched is None:
            group = _Group(m, title_key=t_key, participants=participants)
            _register_group(group)
        else:
            matched.add(m, participants)
            if m.isrc:
                by_isrc.setdefault(m.isrc, matched)

    # Pass 2: match channel reuploads / swapped artist<->title only against canonical Pass-1 groups
    for m in deferred:
        dur = int(m.duration_sec or 0)
        matched = None
        split_dash = split_uploader_title(m.title or "")
        if split_dash:
            left, right = split_dash
            inf_title, inf_parts = _track_participants(right, [left])
            if inf_title and inf_parts:
                matched = _find_match(inf_title, inf_parts, dur, m.isrc, is_reupload=True)
        if matched is None and len(m.artists) == 1 and any(sep in (m.title or "") for sep in (",", "&")):
            # Swapped metadata: artist=["Title"], title="Artist1, Artist2"
            swap_title, swap_parts = _track_participants(m.artists[0], [m.title])
            if swap_title and swap_parts:
                matched = _find_match(swap_title, swap_parts, dur, m.isrc, is_reupload=True)
        if matched is not None:
            matched.add(m, matched.participants)
            continue
        t_key, participants = _track_participants(m.title, m.artists)
        if not t_key:
            continue
        fallback = _find_match(t_key, participants, dur, m.isrc)
        if fallback is not None:
            fallback.add(m, participants)
        else:
            group = _Group(m, title_key=t_key, participants=participants)
            _register_group(group)

    return [{"winner": g.winner, "sources": g.members} for g in ordered_groups]


# ── Albums / Artists ─────────────────────────────────────────────────────────


def _album_years_compatible(m1: Any, m2: Any) -> bool:
    y1 = getattr(m1, "year", None)
    y2 = getattr(m2, "year", None)
    if not y1 or not y2:
        return True
    if y1 == y2:
        return True
    # Different connectors often drift by ±1 year (preorder / TZ release window),
    # while distinct releases with equal connector stay separated when years differ.
    c1 = getattr(m1, "connector", "")
    c2 = getattr(m2, "connector", "")
    if c1 and c2 and c1 != c2 and abs(int(y1) - int(y2)) <= 1:
        tc1 = getattr(m1, "track_count", None)
        tc2 = getattr(m2, "track_count", None)
        if tc1 and tc2 and abs(int(tc1) - int(tc2)) > 2:
            return False
        return True
    return False


def dedup_albums(metas: list) -> list[dict]:
    """Group AlbumMeta by normalized title, compatible artist & release year."""
    by_title: dict[str, list[_Group]] = defaultdict(list)
    ordered_groups: list[_Group] = []
    deferred: list[Any] = []

    def _find_album_match(t_key: str, artist_parts: list[str], meta: Any) -> _Group | None:
        for bkey in _title_lookup_keys(t_key):
            for g in by_title.get(bkey, ()):
                if not artists_compatible(g.participants, artist_parts):
                    continue
                if not all(_album_years_compatible(existing, meta) for existing in g.members):
                    continue
                return g
        return None

    def _register_album(g: _Group) -> None:
        ordered_groups.append(g)
        for bkey in _title_lookup_keys(g.title_key):
            by_title[bkey].append(g)

    for m in metas:
        split_dash = split_uploader_title(m.title or "")
        m_artists = flatten_artist_names([m.artist_name] if m.artist_name else [])
        if split_dash and m_artists:
            left, _right = split_dash
            if not artists_compatible(flatten_artist_names([left]), m_artists, allow_subset=True):
                deferred.append(m)
                continue

        t_key = album_title_key(m.title)
        if not t_key:
            continue
        matched = _find_album_match(t_key, m_artists, m)
        if matched is None:
            _register_album(_Group(m, title_key=t_key, participants=m_artists))
        else:
            matched.add(m, m_artists)

    for m in deferred:
        split_dash = split_uploader_title(m.title or "")
        matched = None
        if split_dash:
            left, right = split_dash
            inf_title = album_title_key(right)
            inf_artists = flatten_artist_names([left])
            if inf_title and inf_artists:
                matched = _find_album_match(inf_title, inf_artists, m)
        if matched is not None:
            matched.add(m, matched.participants)
            continue
        t_key = album_title_key(m.title)
        if not t_key:
            continue
        m_artists = flatten_artist_names([m.artist_name] if m.artist_name else [])
        fallback = _find_album_match(t_key, m_artists, m)
        if fallback is not None:
            fallback.add(m, m_artists)
        else:
            _register_album(_Group(m, title_key=t_key, participants=m_artists))

    out: list[dict] = []
    for g in ordered_groups:
        # prefer ICM as the canonical album catalogue, then richer metadata, then connector
        winner = min(
            g.members,
            key=lambda x: (
                getattr(x, "connector", "") != "icm",
                0 if getattr(x, "cover_url", None) else 1,
                getattr(x, "connector", ""),
            ),
        )
        g.winner = winner
        out.append({"winner": winner, "sources": g.members})
    return out


def _artist_winner_rank(meta: Any) -> tuple[int, int, int, str]:
    name = getattr(meta, "name", "") or ""
    # Prefer ICM, then has_cover, then clean capitalization/script without emojis
    has_non_ascii_decor = 0 if compact_artist_key(name) == dedup_artist_key(name).replace(" ", "") and "(" not in name else 1
    return (
        0 if getattr(meta, "connector", "") == "icm" else 1,
        0 if getattr(meta, "cover_url", None) else 1,
        has_non_ascii_decor,
        getattr(meta, "connector", ""),
    )


def dedup_artists(metas: list) -> list[dict]:
    """Group ArtistMeta across spacing/homoglyph/mixed-script/translit variants.

    Winner: ICM > others (prefers entries with artwork & clean display name).
    """
    ordered_groups: list[_Group] = []
    key_index: dict[str, _Group] = {}

    for m in metas:
        name = getattr(m, "name", "") or ""
        if not compact_artist_key(name):
            continue
        keys = artist_identity_keys(name, allow_translit=True)
        matched = None
        for k in keys:
            matched = key_index.get(k)
            if matched is not None:
                break
        if matched is None:
            g = _Group(m)
            ordered_groups.append(g)
            for k in keys:
                key_index[k] = g
        else:
            matched.add(m)
            for k in keys:
                key_index.setdefault(k, matched)
            if _artist_winner_rank(m) < _artist_winner_rank(matched.winner):
                matched.winner = m

    return [{"winner": g.winner, "sources": g.members} for g in ordered_groups]
