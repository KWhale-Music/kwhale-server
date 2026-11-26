"""Library: atomic finalize (tag -> rename -> row) + integrity checks.

Managed tree layout:
    data/music/{AlbumArtist}/{Album (Year)}/{NN - Title}.{ext}
Service dirs _incoming/, _dropbox/, _failed/ are excluded from orphan scans.
"""

from __future__ import annotations

import hashlib
import logging
import uuid
from pathlib import Path

from .. import db
from ..config import settings
from .tagging import Canonical, fetch_cover, write_tags

log = logging.getLogger("kwhale.library")

# service dirs live at the top level of music_dir
_SERVICE_DIRS = ("_incoming", "_dropbox", "_failed", "_orphaned")


def music_dir() -> Path:
    d = Path(settings.music_dir)
    d.mkdir(parents=True, exist_ok=True)
    return d


def staging_dir() -> Path:
    d = music_dir() / "_incoming"
    d.mkdir(parents=True, exist_ok=True)
    return d


def dropbox_dir() -> Path:
    d = music_dir() / "_dropbox"
    d.mkdir(parents=True, exist_ok=True)
    return d


def failed_dir() -> Path:
    d = music_dir() / "_failed"
    d.mkdir(parents=True, exist_ok=True)
    return d


async def canonical_from_db(track_id: int) -> Canonical | None:
    row = await db.fetch_one("SELECT * FROM tracks WHERE id = :i", {"i": track_id})
    if not row:
        return None
    return Canonical(
        title=row["title"],
        artists=list(row["artists"] or []),
        album=row["album"],
        year=row["year"],
        track_number=row["track_number"],
        track_total=row.get("track_total"),
        disc_number=row.get("disc_number"),
        disc_total=row.get("disc_total"),
        genre=row["genre"],
        cover_url=row["cover_url"],
        isrc=row.get("isrc"),
        explicit=row.get("explicit"),
    )


def _short_hash(track_id: int) -> str:
    return hashlib.md5(str(track_id).encode()).hexdigest()[:8]


async def existing_relative_dirs() -> set[str]:
    """Lowercase set of existing relative paths under the managed tree."""
    out: set[str] = set()
    root = music_dir()
    for p in root.rglob("*"):
        if p.is_file():
            rel = p.relative_to(root).as_posix().lower()
            out.add(rel)
    return out


async def finalize(
    track_id: int,
    tmp_path: Path,
    mime: str,
    cover_bytes: bytes | None = None,
    permanent: bool = False,
) -> Path:
    """Tag staged file with canonical metadata, atomically move into the tree,
    upsert local source row. Returns final path."""
    canonical = await canonical_from_db(track_id)
    if canonical is None:
        raise ValueError(f"track {track_id} not found")

    canonical.track_id_hash = _short_hash(track_id)
    unique_suffix = uuid.uuid4().hex[:8]

    # enrich with track_number from the best source's lazy data (ICM)
    if canonical.track_number is None:
        canonical.track_number = await _lazy_track_number(track_id)

    # lyrics: Musixmatch (synced) -> Genius (plain), best-effort
    if canonical.artists and canonical.title:
        from .providers import get_lyrics

        try:
            got = await get_lyrics(
                canonical.artists[0],
                canonical.title,
                0,
                canonical.album,
            )
            if got:
                canonical.lyrics = got.get("lyrics")
                canonical.lyrics_lrc = got.get("lrc")
        except Exception as e:  # noqa: BLE001 — lyrics never block ingest
            log.debug("lyrics lookup failed: %s", e)

    # cover: fetch if not provided; fall back to iTunes Search
    if cover_bytes is None:
        cover_bytes = await fetch_cover(canonical.cover_url)
    if cover_bytes is None and canonical.artists and canonical.title:
        from .providers import itunes_cover

        try:
            fallback_url = await itunes_cover(
                canonical.artists[0], canonical.title, canonical.album
            )
            if fallback_url:
                cover_bytes = await fetch_cover(fallback_url)
        except Exception as e:  # noqa: BLE001
            log.debug("itunes cover fallback failed: %s", e)

    # tag the staged file (in-place, before it ever enters the tree)
    try:
        write_tags(tmp_path, canonical, cover_bytes)
    except Exception as e:
        log.warning("write_tags failed for track %s: %s", track_id, e)

    # target path: if a different file already occupies the canonical path,
    # keep both (extra copies get a random suffix, never overwrite)
    ext = tmp_path.suffix.lower().lstrip(".") or "mp3"
    existing = await existing_relative_dirs()
    root = music_dir()
    target = root / canonical.file_path(ext, existing)
    if target.exists():
        target = root / canonical.file_path(ext, existing, suffix_override=f"[{unique_suffix}]")
    target.parent.mkdir(parents=True, exist_ok=True)

    # atomic within the same filesystem
    tmp_path.rename(target)
    log.info("library: %s -> %s", tmp_path.name, target)

    # upsert local row
    local_rows = await db.execute_returning(
        """
        INSERT INTO track_sources
            (track_id, connector, external_id, quality_tier, mime, file_path, available,
             permanent, updated_at)
        VALUES (:t, 'local', :ext, 100, :mime, :fp, TRUE, :permanent, NOW())
        ON CONFLICT (connector, external_id) DO UPDATE
        SET file_path = EXCLUDED.file_path,
            mime = EXCLUDED.mime,
            available = TRUE,
            pending_delete_since = NULL,
            permanent = EXCLUDED.permanent,
            track_id = EXCLUDED.track_id,
            updated_at = NOW()
        RETURNING id
        """,
        {
            "t": track_id,
            "ext": f"local::{target.name}",
            "mime": mime,
            "fp": str(target),
            "permanent": permanent,
        },
    )
    local_source_id = int(local_rows[0]["id"]) if local_rows else None
    # queue DSP features, semantic tags, audio and text embeddings for the new local file
    from .rec.queue import DEFAULT_KINDS, enqueue_track

    await enqueue_track(track_id, kinds=DEFAULT_KINDS)

    # local file becomes its own canonical metadata snapshot
    await db.execute(
        """
        INSERT INTO track_source_metadata
            (track_id, connector, external_id, title, artists, album, year,
             track_number, track_total, disc_number, disc_total, genre,
             cover_url, isrc, duration_sec, quality_tier, updated_at)
        VALUES
            (:tid, 'local', :ext, :title, :artists, :album, :year,
             :tn, :tt, :dn, :dt, :genre, :cover, :isrc, NULL, 100, NOW())
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
            quality_tier = EXCLUDED.quality_tier,
            updated_at = NOW()
        """,
        {
            "tid": track_id,
            "ext": f"local::{target.name}",
            "title": canonical.title,
            "artists": canonical.artists,
            "album": canonical.album,
            "year": canonical.year,
            "tn": canonical.track_number,
            "tt": canonical.track_total,
            "dn": canonical.disc_number,
            "dt": canonical.disc_total,
            "genre": canonical.genre,
            "cover": canonical.cover_url,
            "isrc": canonical.isrc,
        },
    )

    # Measure the actual audio header for the admin quality stats. A probe
    # failure is stored in track_quality (tier 0) and never blocks ingest.
    try:
        from . import quality

        await quality.measure_track(track_id, target, source_id=local_source_id)
    except Exception:
        log.warning("quality measure failed for track %s", track_id, exc_info=True)
    return target


async def _lazy_track_number(track_id: int) -> int | None:
    """ICM-only: fetch album track list once per album (24h cache)."""
    src = await db.fetch_one(
        """
        SELECT connector, external_id
        FROM track_sources
        WHERE track_id = :t AND connector = 'icm' AND available
        ORDER BY quality_tier DESC
        LIMIT 1
        """,
        {"t": track_id},
    )
    if not src:
        return None
    from ..connectors import get_connector

    conn = get_connector("icm")
    if conn is None:
        return None
    return await conn.album_track_number_by_source(src["external_id"])


async def _known_local_paths() -> set[str]:
    rows = await db.fetch_all(
        "SELECT file_path FROM track_sources WHERE connector = 'local' AND file_path IS NOT NULL"
    )
    return {r["file_path"] for r in rows}


async def _reenqueue_if_pinned(track_id: int) -> None:
    """Re-download a missing file if the track is favorited or lives in a playlist."""
    pinned = await db.fetch_one(
        "SELECT 1 FROM favorites WHERE track_id = :t LIMIT 1", {"t": track_id}
    )
    if not pinned:
        pinned = await db.fetch_one(
            "SELECT 1 FROM playlist_tracks WHERE track_id = :t LIMIT 1", {"t": track_id}
        )
    if pinned:
        from .downloader import enqueue_if_needed

        await enqueue_if_needed(track_id)


async def fsck() -> dict[str, int]:
    """Integrity pass (daily): broken local rows, orphans, staging purge, empty dirs."""
    stats: dict[str, int] = {}

    rows = await db.fetch_all(
        "SELECT s.id, s.track_id, s.file_path, s.permanent "
        "FROM track_sources s WHERE s.connector = 'local'"
    )
    root = music_dir()

    # Mount guard: many local rows but no files present => the managed tree is
    # likely an unmounted/empty mount. Skip destructive passes entirely rather
    # than dropping every row and orphan-deleting the whole library.
    present = sum(1 for r in rows if r["file_path"] and Path(r["file_path"]).is_file())
    if len(rows) >= 5 and present == 0:
        log.warning(
            "fsck: %d local rows but 0 files present — skipping destructive pass (lost mount?)",
            len(rows),
        )
        stats["broken_rows"] = 0
        stats["orphans"] = 0
        return stats

    # 1. local rows whose file is gone
    dropped = 0
    for row in rows:
        if not row["file_path"] or not Path(row["file_path"]).is_file():
            if row["permanent"]:
                # permanent sources are never auto-dropped: flag and re-download if pinned
                await db.execute(
                    "UPDATE track_sources SET available = FALSE WHERE id = :i", {"i": row["id"]}
                )
                await _reenqueue_if_pinned(row["track_id"])
                continue
            await db.execute("DELETE FROM track_sources WHERE id = :i", {"i": row["id"]})
            dropped += 1
            await _reenqueue_if_pinned(row["track_id"])
    stats["broken_rows"] = dropped

    # 2. orphan scan over the managed tree → quarantine (reversible), never unlink
    known = await _known_local_paths()
    orphan_dir = root / "_orphaned"
    orphans = 0
    for p in root.rglob("*"):
        if not p.is_file() or any(p.is_relative_to(root / d) for d in _SERVICE_DIRS):
            continue
        if str(p) not in known:
            try:
                orphan_dir.mkdir(parents=True, exist_ok=True)
                p.rename(orphan_dir / f"{uuid.uuid4().hex[:8]}_{p.name}")
                orphans += 1
            except OSError as e:
                log.warning("orphan quarantine %s: %s", p, e)
    stats["orphans"] = orphans

    # 3. staging purge (older than 24h)
    from . import gc

    stats.update(await gc.purge_incoming())

    # 4. prune empty dirs (deepest-first)
    import os

    pruned = 0
    for dirpath, _dirnames, _filenames in sorted(
        ((Path(r), d, f) for r, d, f in os.walk(root)), reverse=True
    ):
        p = Path(dirpath)
        if p == root or p.name in _SERVICE_DIRS:
            continue
        try:
            p.rmdir()
            pruned += 1
        except OSError:
            pass
    stats["dirs_pruned"] = pruned
    return stats


async def migrate_flat_files() -> int:
    """One-off: move flat library files (data/music/{ext_id}.ext) into the tree.

    Runs at startup; idempotent — files already in the tree are untouched.
    """
    root = music_dir()
    rows = await db.fetch_all(
        "SELECT track_id, file_path FROM track_sources WHERE connector = 'local'"
    )
    migrated = 0
    for row in rows:
        path = Path(row["file_path"] or "")
        if not path.is_file():
            continue
        if path.parent != root:
            continue  # already inside the tree (or service dir)
        try:
            stage = staging_dir() / path.name
            path.rename(stage)
            mime = (
                "audio/flac"
                if stage.suffix == ".flac"
                else ("audio/mp4" if stage.suffix in (".m4a", ".mp4") else "audio/mpeg")
            )
            await finalize(row["track_id"], stage, mime)
            migrated += 1
        except Exception as e:
            log.warning("migration of %s failed: %s", path, e)
    if migrated:
        log.info("migrated %d flat files into the tree", migrated)
    return migrated
