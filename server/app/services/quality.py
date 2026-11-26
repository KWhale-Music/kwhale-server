"""Measured audio quality of local library files.

The connector `quality_tier` is a promise, not a measurement: local rows are
always tier 100 and a service can resolve to a lower bitrate than advertised.
This module reads the actual file header with mutagen (no decoding) and stores
codec/bitrate/sample-rate/bit-depth/lossless in `track_quality`.

It also owns the admin-facing quality lists and the safe bulk actions:
re-download (force, replace after verification) and delete (quarantine under
music_dir/_orphaned, never unlink).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import shutil
import subprocess
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from mutagen import File as MutagenFile
from mutagen.aiff import AIFF
from mutagen.asf import ASF
from mutagen.flac import FLAC
from mutagen.mp3 import MP3
from mutagen.mp4 import MP4
from mutagen.oggopus import OggOpus
from mutagen.oggvorbis import OggVorbis
from mutagen.wave import WAVE

from .. import db
from ..config import settings

log = logging.getLogger("kwhale.quality")

TIER_UNKNOWN = 0
TIER_LOW = 1       # < 128 kbps
TIER_128 = 2       # 128-191
TIER_192 = 3       # 192-255
TIER_256 = 4       # 256-319
TIER_320 = 5       # 320+
TIER_LOSSLESS = 6

DISTRIBUTION_KEYS = ("lossless", "gte320", "b256", "b192", "b128", "low128", "unknown")


@dataclass(slots=True)
class AudioQuality:
    container: str | None
    codec: str | None
    bitrate_kbps: int | None
    sample_rate: int | None
    bit_depth: int | None
    channels: int | None
    lossless: bool
    tier: int
    bytes: int
    duration_sec: float | None
    error: str | None = None


def tier_for(bitrate_kbps: int | None, lossless: bool) -> int:
    if lossless:
        return TIER_LOSSLESS
    if bitrate_kbps is None:
        return TIER_UNKNOWN
    # LAME/CBR headers report 319/255/191/127 for nominal 320/256/192/128.
    if bitrate_kbps >= 318:
        return TIER_320
    if bitrate_kbps >= 254:
        return TIER_256
    if bitrate_kbps >= 190:
        return TIER_192
    if bitrate_kbps >= 126:
        return TIER_128
    return TIER_LOW


def _int(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _float(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _probe_with_ffprobe(path: Path) -> AudioQuality:
    """Fallback probe for containers mutagen cannot parse (fragmented m4a)."""
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        raise ValueError("unsupported or unreadable audio (no ffprobe)")
    try:
        proc = subprocess.run(  # noqa: S603 — fixed argv, local file
            [
                ffprobe,
                "-v",
                "error",
                "-show_entries",
                "format=duration,bit_rate:stream=codec_type,codec_name,sample_rate,"
                "channels,bits_per_raw_sample,bits_per_sample",
                "-of",
                "json",
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ValueError(f"unsupported or unreadable audio: {exc}") from exc
    if proc.returncode != 0:
        raise ValueError("unsupported or unreadable audio")
    try:
        data = json.loads(proc.stdout or "{}")
    except ValueError as exc:
        raise ValueError("unsupported or unreadable audio") from exc
    streams = [s for s in data.get("streams") or [] if s.get("codec_type") == "audio"]
    if not streams:
        raise ValueError("unsupported or unreadable audio")
    stream = streams[0]
    fmt = data.get("format") or {}
    size = path.stat().st_size
    codec = str(stream.get("codec_name") or path.suffix.lstrip(".") or "unknown").lower()
    sample_rate = _int(stream.get("sample_rate"))
    channels = _int(stream.get("channels"))
    bits = _int(stream.get("bits_per_raw_sample") or stream.get("bits_per_sample"))
    duration = _float(fmt.get("duration"))
    bitrate = _int(fmt.get("bit_rate"))
    if bitrate:
        bitrate //= 1000
    lossless = codec in {"flac", "alac"} or codec.startswith("pcm_")
    if lossless and sample_rate and bits and channels:
        bitrate = round(sample_rate * bits * channels / 1000)
    if not bitrate and duration and duration > 0:
        bitrate = round(size * 8 / duration / 1000)
    return AudioQuality(
        container=path.suffix.lower().lstrip(".") or None,
        codec=codec,
        bitrate_kbps=bitrate,
        sample_rate=sample_rate,
        bit_depth=bits,
        channels=channels,
        lossless=lossless,
        tier=tier_for(bitrate, lossless),
        bytes=size,
        duration_sec=duration,
    )


def probe_file(path: Path) -> AudioQuality:
    """Read an audio header. Raises when the file is empty/unsupported/corrupt."""
    stat = path.stat()
    if stat.st_size <= 0:
        raise ValueError("empty file")
    try:
        audio = MutagenFile(str(path))
    except Exception:  # noqa: BLE001 — fragmented MP4 often trips mutagen
        return _probe_with_ffprobe(path)
    if audio is None or getattr(audio, "info", None) is None:
        return _probe_with_ffprobe(path)

    info = audio.info
    container = path.suffix.lower().lstrip(".") or None
    codec = type(audio).__name__.lower()
    lossless = False
    bitrate = _int(getattr(info, "bitrate", None))
    sample_rate = _int(getattr(info, "sample_rate", None))
    channels = _int(getattr(info, "channels", None))
    bits = _int(getattr(info, "bits_per_sample", None))
    duration = _float(getattr(info, "length", None))

    if isinstance(audio, MP3):
        codec = "mp3"
        if bitrate is not None:
            bitrate //= 1000
    elif isinstance(audio, FLAC):
        codec = "flac"
        lossless = True
    elif isinstance(audio, MP4):
        raw_codec = getattr(info, "codec", None)
        codec = str(raw_codec).lower() if raw_codec else "mp4"
        # .m4a can contain FLAC or ALAC; extension does not tell the truth.
        lossless = codec in {"alac", "flac"}
        if bitrate is not None:
            bitrate //= 1000
    elif isinstance(audio, OggVorbis):
        codec = "vorbis"
        if bitrate is not None:
            bitrate //= 1000
    elif isinstance(audio, OggOpus):
        codec = "opus"
        if bitrate is not None:
            bitrate //= 1000
    elif isinstance(audio, (WAVE, AIFF)):
        codec = "pcm"
        lossless = True
    elif isinstance(audio, ASF):
        codec = "wma"
        if bitrate is not None:
            bitrate //= 1000
    elif bitrate is not None:
        bitrate //= 1000

    if lossless and sample_rate and bits and channels:
        bitrate = round(sample_rate * bits * channels / 1000)
    if not bitrate and duration and duration > 0:
        # VBR/Opus headers sometimes report 0; the file size is honest.
        bitrate = round(stat.st_size * 8 / duration / 1000)
    if bitrate is not None:
        bitrate = max(0, int(bitrate))

    return AudioQuality(
        container=container,
        codec=codec,
        bitrate_kbps=bitrate,
        sample_rate=sample_rate,
        bit_depth=bits,
        channels=channels,
        lossless=lossless,
        tier=tier_for(bitrate, lossless),
        bytes=stat.st_size,
        duration_sec=duration,
    )


async def verify_file(path: str | Path) -> AudioQuality:
    """Probe a staged download before it is allowed to replace a good copy."""
    return await asyncio.to_thread(probe_file, Path(path))


async def _source_id(track_id: int, path: str | Path) -> int | None:
    row = await db.fetch_one(
        """
        SELECT id FROM track_sources
        WHERE connector = 'local' AND track_id = :t AND file_path = :p
        ORDER BY updated_at DESC
        LIMIT 1
        """,
        {"t": track_id, "p": str(path)},
    )
    if row:
        return int(row["id"])
    # The path can be normalised differently by a caller; the track still has
    # exactly one playable local copy in the normal case.
    row = await db.fetch_one(
        """
        SELECT id FROM track_sources
        WHERE connector = 'local' AND track_id = :t
        ORDER BY updated_at DESC
        LIMIT 1
        """,
        {"t": track_id},
    )
    return int(row["id"]) if row else None


async def measure_track(
    track_id: int,
    path: str | Path,
    source_id: int | None = None,
) -> AudioQuality | None:
    """Measure one local file and upsert its row. Probe errors become tier 0."""
    file_path = Path(path)
    quality: AudioQuality | None
    error: str | None = None
    try:
        quality = await asyncio.to_thread(probe_file, file_path)
    except Exception as exc:  # noqa: BLE001 — a corrupt file must not break ingest
        quality = None
        error = str(exc)[:500]
        log.warning("quality probe failed for track %s (%s): %s", track_id, file_path, exc)

    if source_id is None:
        source_id = await _source_id(track_id, file_path)

    size = 0
    try:
        size = file_path.stat().st_size
    except OSError:
        pass

    await db.execute(
        """
        INSERT INTO track_quality
            (track_id, local_source_id, file_path, container, codec, bitrate_kbps,
             sample_rate, bit_depth, channels, lossless, tier, bytes, duration_sec,
             measured_at, error)
        VALUES
            (:track_id, :source_id, :file_path, :container, :codec, :bitrate,
             :sample_rate, :bit_depth, :channels, :lossless, :tier, :bytes,
             :duration, NOW(), :error)
        ON CONFLICT (track_id) DO UPDATE SET
            local_source_id = EXCLUDED.local_source_id,
            file_path = EXCLUDED.file_path,
            container = EXCLUDED.container,
            codec = EXCLUDED.codec,
            bitrate_kbps = EXCLUDED.bitrate_kbps,
            sample_rate = EXCLUDED.sample_rate,
            bit_depth = EXCLUDED.bit_depth,
            channels = EXCLUDED.channels,
            lossless = EXCLUDED.lossless,
            tier = EXCLUDED.tier,
            bytes = EXCLUDED.bytes,
            duration_sec = EXCLUDED.duration_sec,
            measured_at = NOW(),
            error = EXCLUDED.error
        """,
        {
            "track_id": track_id,
            "source_id": source_id,
            "file_path": str(file_path),
            "container": quality.container if quality else None,
            "codec": quality.codec if quality else None,
            "bitrate": quality.bitrate_kbps if quality else None,
            "sample_rate": quality.sample_rate if quality else None,
            "bit_depth": quality.bit_depth if quality else None,
            "channels": quality.channels if quality else None,
            "lossless": bool(quality and quality.lossless),
            "tier": quality.tier if quality else TIER_UNKNOWN,
            "bytes": quality.bytes if quality else size,
            "duration": quality.duration_sec if quality else None,
            "error": error,
        },
    )
    return quality


def _candidate_where(force: bool, track_ids: list[int] | None) -> tuple[str, dict[str, Any]]:
    clauses = [
        "s.connector = 'local'",
        "s.file_path IS NOT NULL",
        "s.available",
    ]
    params: dict[str, Any] = {"force": force}
    if not force:
        clauses.append(
            "(q.track_id IS NULL OR q.measured_at < s.updated_at)"
        )
    if track_ids:
        names = []
        for index, track_id in enumerate(track_ids):
            name = f"tid{index}"
            names.append(f":{name}")
            params[name] = int(track_id)
        clauses.append(f"s.track_id IN ({', '.join(names)})")
    return " AND ".join(clauses), params


async def _count_candidates(force: bool, track_ids: list[int] | None) -> int:
    where, params = _candidate_where(force, track_ids)
    row = await db.fetch_one(
        f"""
        SELECT count(DISTINCT s.track_id) AS n
        FROM track_sources s
        LEFT JOIN track_quality q ON q.track_id = s.track_id
        WHERE {where}
        """,
        params,
    )
    return int(row["n"]) if row else 0


async def scan(
    *,
    limit: int | None = None,
    force: bool = False,
    track_ids: list[int] | None = None,
    on_progress: Callable[[int, int, int], None] | None = None,
) -> dict[str, Any]:
    """Measure local files that are missing a row or changed since measurement."""
    where, params = _candidate_where(force, track_ids)
    total = await _count_candidates(force, track_ids)
    params["limit"] = limit if limit and limit > 0 else 1_000_000
    rows = await db.fetch_all(
        f"""
        SELECT DISTINCT ON (s.track_id)
               s.id AS source_id, s.track_id, s.file_path
        FROM track_sources s
        LEFT JOIN track_quality q ON q.track_id = s.track_id
        WHERE {where}
        ORDER BY s.track_id, s.updated_at DESC
        LIMIT :limit
        """,
        params,
    )
    if on_progress:
        on_progress(0, total, 0)
    scanned = 0
    errors = 0
    for index, row in enumerate(rows, start=1):
        quality = await measure_track(
            int(row["track_id"]),
            row["file_path"],
            source_id=int(row["source_id"]),
        )
        scanned += 1
        if quality is None:
            errors += 1
        if on_progress:
            on_progress(scanned, total, errors)
        if index % 100 == 0:
            log.info("quality scan: %s/%s (errors %s)", index, total, errors)
    return {"scanned": scanned, "total": total, "errors": errors}


_SCAN_STATE: dict[str, Any] = {
    "running": False,
    "scanned": 0,
    "total": 0,
    "errors": 0,
    "startedAt": None,
    "finishedAt": None,
}
_SCAN_LOCK = asyncio.Lock()


def scan_status() -> dict[str, Any]:
    return dict(_SCAN_STATE)


async def start_scan(
    *,
    limit: int | None = None,
    force: bool = False,
    track_ids: list[int] | None = None,
) -> bool:
    if _SCAN_LOCK.locked() or _SCAN_STATE["running"]:
        return False
    # Mark it running before the task is scheduled, otherwise an immediate
    # status poll sees running=false and stops polling.
    _SCAN_STATE.update(
        running=True,
        scanned=0,
        total=0,
        errors=0,
        startedAt=datetime.now(UTC).isoformat(),
        finishedAt=None,
    )
    asyncio.create_task(_run_scan(limit=limit, force=force, track_ids=track_ids))
    return True


async def _run_scan(
    *,
    limit: int | None,
    force: bool,
    track_ids: list[int] | None,
) -> None:
    async with _SCAN_LOCK:
        _SCAN_STATE.update(
            running=True,
            scanned=0,
            total=0,
            errors=0,
            startedAt=datetime.now(UTC).isoformat(),
            finishedAt=None,
        )

        def progress(scanned: int, total: int, errors: int) -> None:
            _SCAN_STATE.update(scanned=scanned, total=total, errors=errors)

        try:
            await scan(
                limit=limit,
                force=force,
                track_ids=track_ids,
                on_progress=progress,
            )
        except Exception:
            log.exception("quality scan failed")
        finally:
            _SCAN_STATE.update(
                running=False,
                finishedAt=datetime.now(UTC).isoformat(),
            )


def _shape_stats(row: dict[str, Any], *, with_bytes: bool = False) -> dict[str, Any]:
    tracks = int(row.get("tracks") or 0)
    measured = int(row.get("measured") or 0)
    lossless = int(row.get("lossless") or 0)
    avg = row.get("avg_lossy")
    distribution = {
        "lossless": int(row.get("d_lossless") or 0),
        "gte320": int(row.get("d_gte320") or 0),
        "b256": int(row.get("d_b256") or 0),
        "b192": int(row.get("d_b192") or 0),
        "b128": int(row.get("d_b128") or 0),
        "low128": int(row.get("d_low128") or 0),
        "unknown": int(row.get("d_unknown") or 0),
    }
    shaped = {
        "tracks": tracks,
        "measured": measured,
        "unknown": max(0, tracks - measured),
        "lossless": lossless,
        "losslessShare": round(lossless / measured, 3) if measured else 0.0,
        "avgLossyBitrateKbps": round(float(avg), 1) if avg is not None else None,
        "below192": int(row.get("below_192") or 0),
        "distribution": distribution,
    }
    if with_bytes:
        shaped["bytes"] = int(row.get("bytes") or 0)
    return shaped


_STATS_COLUMNS = """
    count(*) AS tracks,
    count(q.track_id) FILTER (WHERE q.error IS NULL AND q.tier > 0) AS measured,
    count(*) FILTER (WHERE q.tier = 6 AND q.error IS NULL) AS lossless,
    avg(q.bitrate_kbps) FILTER (WHERE q.tier BETWEEN 1 AND 5 AND q.error IS NULL) AS avg_lossy,
    count(*) FILTER (WHERE q.tier IN (1, 2) AND q.error IS NULL) AS below_192,
    {%BYTES%}
    count(*) FILTER (WHERE q.tier = 6 AND q.error IS NULL) AS d_lossless,
    count(*) FILTER (WHERE q.tier = 5 AND q.error IS NULL) AS d_gte320,
    count(*) FILTER (WHERE q.tier = 4 AND q.error IS NULL) AS d_b256,
    count(*) FILTER (WHERE q.tier = 3 AND q.error IS NULL) AS d_b192,
    count(*) FILTER (WHERE q.tier = 2 AND q.error IS NULL) AS d_b128,
    count(*) FILTER (WHERE q.tier = 1 AND q.error IS NULL) AS d_low128,
    count(*) FILTER (
        WHERE q.track_id IS NULL OR q.error IS NOT NULL OR q.tier = 0
    ) AS d_unknown
"""


async def overview(days: int = 30) -> dict[str, Any]:
    days = max(0, int(days))
    # days=0 means "all time" in the UI; make_interval(0) would mean "now".
    window_days = days if days > 0 else 36500
    columns = _STATS_COLUMNS.replace("{%BYTES%}", "coalesce(sum(q.bytes), 0) AS bytes,")
    local_cte = """
        WITH local AS (
            SELECT DISTINCT ON (track_id) id, track_id, updated_at
            FROM track_sources
            WHERE connector = 'local' AND file_path IS NOT NULL AND available
            ORDER BY track_id, updated_at DESC
        )
    """
    library_row = await db.fetch_one(
        local_cte
        + f"""
        SELECT {columns}
        FROM local l
        LEFT JOIN track_quality q ON q.track_id = l.track_id
        """,
    )
    downloaded_row = await db.fetch_one(
        local_cte
        + f"""
        SELECT {columns}
        FROM local l
        LEFT JOIN track_quality q ON q.track_id = l.track_id
        WHERE l.updated_at >= NOW() - make_interval(days => :days)
        """,
        {"days": window_days},
    )
    played_columns = _STATS_COLUMNS.replace("{%BYTES%}", "")
    played_row = await db.fetch_one(
        f"""
        WITH played AS (
            SELECT DISTINCT track_id
            FROM events
            WHERE type = 'play' AND track_id IS NOT NULL
              AND COALESCE(occurred_at, received_at) >= NOW() - make_interval(days => :days)
        )
        SELECT {played_columns}
        FROM played p
        LEFT JOIN track_quality q ON q.track_id = p.track_id
        """,
        {"days": window_days},
    )
    played = _shape_stats(played_row or {})
    played["estimated"] = played["unknown"]
    return {
        "days": days,
        "library": _shape_stats(library_row or {}, with_bytes=True),
        "downloaded": _shape_stats(downloaded_row or {}, with_bytes=True),
        "played": played,
        "generatedAt": datetime.now(UTC).isoformat(),
    }


def _current_equiv(
    tier: int | None,
    lossless: bool | None,
    bit_depth: int | None,
    bitrate_kbps: int | None,
) -> int | None:
    if not tier:
        return None
    if lossless:
        return 95 if (bit_depth or 16) >= 24 else 85
    bitrate = bitrate_kbps or 0
    if bitrate >= 318:
        return 65
    if bitrate >= 254:
        return 60
    if bitrate >= 190:
        return 55
    if bitrate >= 126:
        return 50
    return 40


def _source_label(connector: str | None, tier: int | None) -> str:
    level = "неизвестно"
    if tier is not None:
        if tier >= 95:
            level = "Hi-Res FLAC"
        elif tier >= 85:
            level = "FLAC"
        elif tier >= 65:
            level = "MP3 320"
        elif tier >= 50:
            level = "MP3/AAC 128–256"
        elif tier > 0:
            level = "низкое"
    return f"{level} ({connector})" if connector else level


def _iso(value: Any) -> str | None:
    if value is None:
        return None
    return value.isoformat() if hasattr(value, "isoformat") else str(value)


async def list_tracks(
    *,
    days: int = 30,
    limit: int = 50,
    offset: int = 0,
    max_bitrate: int | None = None,
    min_bitrate: int | None = None,
    lossless: bool | None = None,
    codec: str | None = None,
    tier: int | None = None,
    query: str | None = None,
    played: bool | None = None,
    downloaded: bool | None = None,
    sort: str = "worst",
) -> dict[str, Any]:
    days = max(0, int(days))
    window_days = days if days > 0 else 36500
    limit = max(1, min(200, int(limit)))
    offset = max(0, int(offset))
    clauses: list[str] = []
    params: dict[str, Any] = {"days": window_days, "limit": limit, "offset": offset}

    if max_bitrate is not None:
        clauses.append("q.bitrate_kbps IS NOT NULL AND q.bitrate_kbps <= :max_bitrate")
        params["max_bitrate"] = int(max_bitrate)
    if min_bitrate is not None:
        clauses.append("q.bitrate_kbps IS NOT NULL AND q.bitrate_kbps >= :min_bitrate")
        params["min_bitrate"] = int(min_bitrate)
    if lossless is not None:
        clauses.append("q.lossless = :lossless")
        params["lossless"] = bool(lossless)
    if codec:
        clauses.append("lower(coalesce(q.codec, '')) = :codec")
        params["codec"] = codec.lower()
    if tier is not None:
        if int(tier) == TIER_UNKNOWN:
            clauses.append(
                "(q.track_id IS NULL OR q.error IS NOT NULL OR q.tier = 0)"
            )
        else:
            clauses.append("q.tier = :tier")
            params["tier"] = int(tier)
    if query:
        clauses.append(
            "(t.title ILIKE :query OR t.artists_text ILIKE :query)"
        )
        params["query"] = f"%{query.strip()}%"
    if played is True:
        clauses.append("ev.track_id IS NOT NULL")
    elif played is False:
        clauses.append("ev.track_id IS NULL")
    if downloaded is True:
        clauses.append("l.updated_at >= NOW() - make_interval(days => :days)")

    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    order = {
        "worst": "q.tier ASC NULLS LAST, q.bitrate_kbps ASC NULLS LAST, ev.plays DESC NULLS LAST",
        "bitrate_asc": "q.bitrate_kbps ASC NULLS LAST",
        "bitrate_desc": "q.bitrate_kbps DESC NULLS LAST",
        "plays_desc": "ev.plays DESC NULLS LAST",
        "downloaded_desc": "l.updated_at DESC",
        "title": "t.title ASC",
    }.get(sort, "q.tier ASC NULLS LAST, q.bitrate_kbps ASC NULLS LAST")

    rows = await db.fetch_all(
        f"""
        WITH local AS (
            SELECT DISTINCT ON (track_id) id, track_id, file_path, updated_at, permanent
            FROM track_sources
            WHERE connector = 'local' AND file_path IS NOT NULL AND available
            ORDER BY track_id, updated_at DESC
        ),
        plays AS (
            SELECT track_id, count(*) AS plays,
                   max(COALESCE(occurred_at, received_at)) AS last_played
            FROM events
            WHERE type = 'play' AND track_id IS NOT NULL
              AND COALESCE(occurred_at, received_at) >= NOW() - make_interval(days => :days)
            GROUP BY track_id
        )
        SELECT t.id, t.title, t.artists, t.album, t.duration_sec,
               l.id AS source_id, l.file_path, l.updated_at, l.permanent,
               q.container, q.codec, q.bitrate_kbps, q.sample_rate, q.bit_depth,
               q.channels, q.lossless, q.tier, q.measured_at, q.error,
               ev.plays, ev.last_played,
               best.connector AS best_connector, best.quality_tier AS best_tier,
               count(*) OVER() AS total
        FROM local l
        JOIN tracks t ON t.id = l.track_id
        LEFT JOIN track_quality q ON q.track_id = l.track_id
        LEFT JOIN plays ev ON ev.track_id = l.track_id
        LEFT JOIN LATERAL (
            SELECT s2.connector, s2.quality_tier
            FROM track_sources s2
            WHERE s2.track_id = l.track_id
              AND s2.connector != 'local' AND s2.available
            ORDER BY s2.quality_tier DESC, s2.connector
            LIMIT 1
        ) best ON TRUE
        {where}
        ORDER BY {order}
        LIMIT :limit OFFSET :offset
        """,
        params,
    )

    items: list[dict[str, Any]] = []
    total = 0
    for row in rows:
        total = int(row.get("total") or 0)
        current_equiv = _current_equiv(
            row.get("tier"),
            row.get("lossless"),
            row.get("bit_depth"),
            row.get("bitrate_kbps"),
        )
        best_tier = row.get("best_tier")
        permanent = bool(row.get("permanent"))
        can_upgrade = bool(
            not permanent
            and best_tier is not None
            and int(best_tier) > 0
            and (current_equiv is None or int(best_tier) > current_equiv)
        )
        quality = None
        if row.get("codec") is not None or row.get("tier") is not None:
            quality = {
                "container": row.get("container"),
                "codec": row.get("codec"),
                "bitrateKbps": row.get("bitrate_kbps"),
                "sampleRate": row.get("sample_rate"),
                "bitDepth": row.get("bit_depth"),
                "channels": row.get("channels"),
                "lossless": bool(row.get("lossless")),
                "tier": int(row.get("tier") or 0),
                "measuredAt": _iso(row.get("measured_at")),
                "error": row.get("error"),
            }
        items.append(
            {
                "id": str(row["id"]),
                "title": row["title"],
                "artists": list(row.get("artists") or []),
                "album": row.get("album"),
                "durationSec": row.get("duration_sec"),
                "filePath": row.get("file_path"),
                "permanent": permanent,
                "downloadedAt": _iso(row.get("updated_at")),
                "quality": quality,
                "plays": int(row.get("plays") or 0),
                "lastPlayedAt": _iso(row.get("last_played")),
                "best": (
                    {
                        "connector": row.get("best_connector"),
                        "tier": int(best_tier),
                        "label": _source_label(row.get("best_connector"), int(best_tier)),
                    }
                    if best_tier is not None
                    else None
                ),
                "canUpgrade": can_upgrade,
            }
        )
    return {"items": items, "total": total, "limit": limit, "offset": offset}


async def _drop_orphan_quality(track_id: int) -> None:
    await db.execute(
        """
        DELETE FROM track_quality q
        WHERE q.track_id = :t
          AND NOT EXISTS (
              SELECT 1 FROM track_sources s
              WHERE s.track_id = :t AND s.connector = 'local'
          )
        """,
        {"t": track_id},
    )


async def _quarantine_file(track_id: int, path: Path, prefix: str) -> int:
    root = Path(settings.music_dir) / "_orphaned" / "quality"
    root.mkdir(parents=True, exist_ok=True)
    destination = root / f"{prefix}-{track_id}-{uuid.uuid4().hex[:8]}-{path.name}"
    size = path.stat().st_size if path.is_file() else 0
    shutil.move(str(path), str(destination))
    return size


async def quarantine_old_copies(track_id: int, keep_path: str | Path) -> dict[str, int]:
    """Remove other local rows/files after a force re-download.

    The new file is already finalized and measured; old files are moved to
    _orphaned (reversible), never unlinked.
    """
    keep = str(keep_path)
    rows = await db.fetch_all(
        """
        SELECT id, file_path, permanent
        FROM track_sources
        WHERE track_id = :t AND connector = 'local' AND file_path != :keep
        """,
        {"t": track_id, "keep": keep},
    )
    moved = 0
    bytes_moved = 0
    for row in rows:
        if row["permanent"]:
            continue
        path = Path(row["file_path"]) if row["file_path"] else None
        try:
            if path and path.is_file():
                size = path.stat().st_size
                await _quarantine_file(track_id, path, "replaced")
                bytes_moved += size
                moved += 1
        except Exception as exc:  # noqa: BLE001 — the new copy is already in place
            log.warning("quality replace: could not quarantine %s: %s", path, exc)
        # The new file is finalized and measured; the old row must not remain
        # playable even if moving its file failed (fsck will quarantine it).
        try:
            await db.execute("DELETE FROM track_sources WHERE id = :i", {"i": row["id"]})
        except Exception as exc:  # noqa: BLE001
            log.warning("quality replace: could not delete row %s: %s", row["id"], exc)
    await _drop_orphan_quality(track_id)
    return {"moved": moved, "bytes": bytes_moved}


async def delete_local(track_ids: list[int]) -> dict[str, Any]:
    deleted_files = 0
    bytes_freed = 0
    skipped: list[dict[str, str]] = []
    for track_id in dict.fromkeys(int(v) for v in track_ids):
        rows = await db.fetch_all(
            """
            SELECT id, file_path, permanent
            FROM track_sources
            WHERE track_id = :t AND connector = 'local'
            """,
            {"t": track_id},
        )
        if not rows:
            skipped.append({"trackId": str(track_id), "reason": "нет локального файла"})
            continue
        for row in rows:
            if row["permanent"]:
                skipped.append(
                    {"trackId": str(track_id), "reason": "загруженный пользователем файл защищён"}
                )
                continue
            path = Path(row["file_path"]) if row["file_path"] else None
            try:
                if path and path.is_file():
                    size = path.stat().st_size
                    await _quarantine_file(track_id, path, "deleted")
                    bytes_freed += size
                    deleted_files += 1
                await db.execute("DELETE FROM track_sources WHERE id = :i", {"i": row["id"]})
            except Exception as exc:  # noqa: BLE001
                log.warning("quality delete: %s: %s", path, exc)
                skipped.append({"trackId": str(track_id), "reason": str(exc)})
        await _drop_orphan_quality(track_id)
    return {
        "deletedFiles": deleted_files,
        "bytesFreed": bytes_freed,
        "skipped": skipped,
    }


async def redownload(track_ids: list[int]) -> dict[str, Any]:
    from . import downloader

    queued = 0
    skipped: list[dict[str, str]] = []
    for track_id in dict.fromkeys(int(v) for v in track_ids):
        local = await db.fetch_all(
            """
            SELECT permanent FROM track_sources
            WHERE track_id = :t AND connector = 'local'
            """,
            {"t": track_id},
        )
        if local and any(row["permanent"] for row in local):
            skipped.append(
                {"trackId": str(track_id), "reason": "загруженный пользователем файл защищён"}
            )
            continue
        job_id = await downloader.enqueue(track_id, force=True)
        if job_id is None:
            skipped.append({"trackId": str(track_id), "reason": "уже в очереди"})
        else:
            queued += 1
    return {"queued": queued, "skipped": skipped}


async def _cli() -> None:
    parser = argparse.ArgumentParser(description="KWhale local audio quality scanner")
    parser.add_argument("--scan", action="store_true", help="measure missing files (default)")
    parser.add_argument("--force", action="store_true", help="re-measure every local file")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--track-id", type=int, action="append", default=None)
    parser.add_argument(
        "--redownload-corrupt",
        action="store_true",
        help="queue fresh copies for DSP jobs that failed as corrupt",
    )
    args = parser.parse_args()

    if args.redownload_corrupt:
        rows = await db.fetch_all(
            """
            SELECT DISTINCT track_id
            FROM index_jobs
            WHERE kind = 'dsp' AND status = 'failed'
              AND last_error LIKE 'corrupt:%'
              AND track_id IN (
                  SELECT track_id FROM track_sources WHERE connector = 'local'
              )
            ORDER BY track_id
            LIMIT :l
            """,
            {"l": args.limit or 100},
        )
        ids = [int(r["track_id"]) for r in rows]
        result = await redownload(ids)
        print(json.dumps({"candidates": len(ids), **result}, ensure_ascii=False))
        await db.close_engine()
        return

    stats = await scan(limit=args.limit, force=args.force, track_ids=args.track_id)
    print(json.dumps(stats, ensure_ascii=False))
    await db.close_engine()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(_cli())
