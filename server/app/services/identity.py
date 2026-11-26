"""Track identity: one recording, one key — regardless of which source says it.

The library used to key likes strictly by the tracks-table row id. The same
recording legitimately arrives as several rows (a source with a collab string
"Тимати,MOT,ЕГОР КРИД" vs a source with three separate names; "(feat. X)" in
the title vs array; a duration 2-15s off), so a like on one row showed as
«not liked» on its twin and liking the twin added a duplicate to the library.

Identity rule (shared with dedup.py & merge.py):
  identity_hash = sha1(normalized clean title | compact PRIMARY artist)
  same recording = equal hash AND duration_close(...) (or compatible collab)
"""

from __future__ import annotations

import hashlib

from .dedup import duration_close as _dedup_duration_close
from .dedup import normalize_title_or_raw
from .normalize import (
    artists_compatible,
    compact_artist_key,
    extract_featured_artists,
    flatten_artist_names,
    split_uploader_title,
)

DURATION_TOLERANCE_SEC = 15


def _clean_title_and_participants(
    title: str | None, artists: list[str] | None
) -> tuple[str, list[str]]:
    raw_title = title or ""
    raw_artists = [a for a in (artists or []) if a and a.strip()]
    split_dash = split_uploader_title(raw_title)
    if split_dash and raw_artists:
        left, right = split_dash
        if artists_compatible(flatten_artist_names([left]), raw_artists, allow_subset=True):
            raw_title = right
    clean_title, featured = extract_featured_artists(raw_title)
    participants = flatten_artist_names(raw_artists)
    for f in featured:
        if f not in participants:
            participants.append(f)
    return normalize_title_or_raw(clean_title), participants


def primary_artist_key(artists: list[str] | None) -> str:
    flat = flatten_artist_names(artists)
    return compact_artist_key(flat[0]) if flat else ""


def identity_hash(title: str | None, artists: list[str] | None) -> str:
    """Stable hash of (clean title, primary artist). Both sides normalized."""
    title_key, participants = _clean_title_and_participants(title, artists)
    artist_key = compact_artist_key(participants[0]) if participants else ""
    raw = f"{title_key}|{artist_key}".encode()
    return hashlib.sha1(raw).hexdigest()[:32]


def duration_close(
    a: int | None, b: int | None, tolerance: int = DURATION_TOLERANCE_SEC
) -> bool:
    return _dedup_duration_close(int(a or 0), int(b or 0), tolerance=tolerance)


def same_recording(
    title_a: str | None,
    artists_a: list[str] | None,
    duration_a: int | None,
    title_b: str | None,
    artists_b: list[str] | None,
    duration_b: int | None,
) -> bool:
    tk_a, parts_a = _clean_title_and_participants(title_a, artists_a)
    tk_b, parts_b = _clean_title_and_participants(title_b, artists_b)
    if not tk_a or tk_a != tk_b:
        return False
    if not artists_compatible(parts_a, parts_b):
        return False
    return duration_close(duration_a, duration_b)
