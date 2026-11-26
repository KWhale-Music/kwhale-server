"""Manual track pipeline: fingerprint -> upsert/merge -> finalize.

Used by both POST /v1/uploads/tracks and the _dropbox/ watcher. A manual file that
cannot be resolved (no usable tags, no Shazam/AcoustID match) goes to
_failed/ quarantine (never auto-deleted).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from .. import db
from . import fingerprint
from .genres import normalize_genres
from .identity import identity_hash
from .library import failed_dir, finalize, staging_dir

log = logging.getLogger("kwhale.manual")

_AUDIO_EXTS = {".mp3", ".flac", ".m4a", ".mp4", ".ogg", ".opus", ".wav", ".aac"}


def is_audio(path: Path) -> bool:
    return path.suffix.lower() in _AUDIO_EXTS


async def process_manual_file(
    src: Path, hint: dict[str, Any] | None = None, user_id: int | None = None
) -> int | None:
    """Full pipeline for one manual file. Returns track_id or None (quarantined).

    src may live in _dropbox/ or _incoming/manual/; the file is consumed
    (moved) by this call.
    """
    if not is_audio(src):
        log.warning("manual: %s is not audio — moved to _failed", src.name)
        _quarantine(src)
        return None

    # 1. resolve canonical metadata: hints > existing tags > shazam > acoustid
    meta = await fingerprint.resolve(src, hint=hint)
    if not meta or not meta.get("title") or not meta.get("artist"):
        log.warning("manual: %s unresolved — moved to _failed", src.name)
        _quarantine(src)
        return None

    title = str(meta.get("title") or "")
    artist = str(meta.get("artist") or "")
    album = str(meta.get("album") or "") or None
    year = _to_int(meta.get("year"))
    track_number = _to_int(meta.get("track_number"))
    genre = str(meta.get("genre") or "") or None
    duration = _to_int(meta.get("duration"))
    cover_url = meta.get("cover_url") or None
    isrc = str(meta.get("isrc") or "") or None

    # 2. dedup vs existing library: fuzzy title+artist+duration ±2s
    track_id = await _match_existing(title, artist, duration)
    if track_id is None:
        manual_artists = [a.strip() for a in artist.split(",") if a.strip()] or ["Unknown"]
        rows = await db.execute_returning(
            """
            INSERT INTO tracks
                (title, artists, album, duration_sec, isrc, cover_url, year, track_number, genre, identity_hash, last_seen_at)
            VALUES (:t, :a, :al, :d, :isrc, :c, :y, :tn, :g, :ih, NOW())
            RETURNING id
            """,
            {
                "t": title,
                "a": manual_artists,
                "al": album,
                "d": duration,
                "isrc": isrc,
                "c": cover_url,
                "y": year,
                "tn": track_number,
                "g": genre,
                "ih": identity_hash(title, manual_artists),
            },
        )
        track_id = rows[0]["id"]
        log.info("manual: new track %s (%s)", track_id, title)
    else:
        # existing track: enrich missing fields with fingerprint data
        await db.execute(
            """
            UPDATE tracks SET
                album = COALESCE(album, :al),
                year = COALESCE(year, :y),
                track_number = COALESCE(track_number, :tn),
                genre = COALESCE(genre, :g),
                genres_norm = COALESCE(:gn, genres_norm),
                cover_url = COALESCE(cover_url, :c),
                last_seen_at = NOW()
            WHERE id = :t
            """,
            {"al": album, "y": year, "tn": track_number, "g": genre,
             "gn": normalize_genres([genre]) or None,
             "c": cover_url, "t": track_id},
        )
        log.info("manual: matched existing track %s for %s", track_id, title)

    # 3. stage + finalize (tags from canonical DB row, move to tree)
    stage = staging_dir() / f"manual_{src.name}"
    src.rename(stage)
    try:
        from .stream_service import mime_for_path

        mime = mime_for_path(stage)
        final = await finalize(track_id, stage, mime, permanent=True)
        log.info("manual: %s -> %s", src.name, final)
    except Exception:
        # finalize failed: park the file in _failed so nothing is lost
        _quarantine(stage)
        raise

    # 4. auto-like for the uploading user (dropbox has no user — pinned by file)
    if user_id:
        await db.execute(
            "INSERT INTO favorites (user_id, track_id) VALUES (:u, :t) ON CONFLICT DO NOTHING",
            {"u": user_id, "t": track_id},
        )
    return track_id


async def _match_existing(title: str, artist: str, duration: int | None) -> int | None:
    """Fuzzy match against tracks: normalized title + ANY artist overlap (±2s).

    Artist strings may contain multiple comma-separated names on either side;
    a match is any pairwise overlap of their normalized name parts.
    """
    from .dedup import normalize_artist, normalize_title

    nt = normalize_title(title)
    if not nt:
        return None
    wanted = {normalize_artist(a) for a in artist.split(",") if normalize_artist(a)}
    if not wanted:
        return None

    rows = await db.fetch_all("SELECT id, title, artists, duration_sec FROM tracks LIMIT 50000")
    for r in rows:
        if normalize_title(r["title"]) != nt:
            continue
        have = {normalize_artist(a) for a in (r["artists"] or []) if a}
        if not wanted & have:
            continue
        if duration and r["duration_sec"] and abs(duration - r["duration_sec"]) > 2:
            continue
        return r["id"]
    return None


def _to_int(v: Any) -> int | None:
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _quarantine(src: Path) -> None:
    try:
        failed_dir().mkdir(parents=True, exist_ok=True)
        dst = failed_dir() / src.name
        if src.is_file():
            if dst.exists():
                dst.unlink()
            src.rename(dst)
    except OSError as e:
        log.error("quarantine %s failed: %s", src, e)
