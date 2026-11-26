"""Downloader: favorites-driven download queue.

One job per track (unique), asyncio-driven, state in Postgres so restarts
resume pending work. Downloaded file -> tagged -> local source row with
pending_delete_since=NULL. Unlike-grace (14d) handled by GC.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

from .. import db
from ..connectors import get_connector

log = logging.getLogger("kwhale.downloader")

MAX_ATTEMPTS = 3
_wake = asyncio.Event()
_runner_started = False


def staging_dir() -> Path:
    """Staging zone for connector downloads (GC purges files older than 24h)."""
    from .library import staging_dir

    return staging_dir()


async def enqueue(
    track_id: int,
    target_type: str | None = None,
    target_id: int | None = None,
    *,
    force: bool = False,
) -> int | None:
    """Create a download job. Returns the job id, or None when nothing was queued.

    None means: the track already has a local file, or a job is already
    pending/running. ``target_type``/``target_id`` record what the user asked
    for (track/album/playlist) so the client can show bulk progress.

    ``force`` is for the admin quality flow: download again from the best
    source, verify the staged file, finalize it, then quarantine the old local
    copy. User uploads (permanent) are never replaced.
    """
    local_rows = await db.fetch_all(
        "SELECT id, permanent FROM track_sources WHERE track_id = :t AND connector = 'local'",
        {"t": track_id},
    )
    if local_rows:
        if not force or any(row.get("permanent") for row in local_rows):
            return None  # shared library / protected user upload
    rows = await db.execute_returning(
        """
        INSERT INTO download_jobs (track_id, status, target_type, target_id, replace_local)
        VALUES (:t, 'pending', :tt, :ti, :force)
        ON CONFLICT (track_id) DO UPDATE
        SET status = 'pending', error = NULL, attempts = 0, updated_at = NOW(),
            target_type = COALESCE(EXCLUDED.target_type, download_jobs.target_type),
            target_id = COALESCE(EXCLUDED.target_id, download_jobs.target_id),
            replace_local = EXCLUDED.replace_local
        WHERE download_jobs.status NOT IN ('pending', 'running')
        RETURNING id
        """,
        {"t": track_id, "tt": target_type, "ti": target_id, "force": force},
    )
    if not rows:
        return None
    _wake.set()
    return rows[0]["id"]


async def enqueue_if_needed(track_id: int) -> None:
    """Create download job if the track has no local file and no pending/running one."""
    await enqueue(track_id)


async def _next_job() -> dict[str, Any] | None:
    return await db.fetch_one(
        """
        SELECT j.id, j.track_id, j.attempts, j.replace_local
        FROM download_jobs j
        WHERE j.status = 'pending'
        ORDER BY j.id
        LIMIT 1
        FOR UPDATE SKIP LOCKED
        """,
    )


async def _sources_for(track_id: int) -> list[dict[str, Any]]:
    rows = await db.fetch_all(
        """
        SELECT connector, external_id, quality_tier
        FROM track_sources
        WHERE track_id = :t AND connector != 'local' AND available
        ORDER BY quality_tier DESC
        """,
        {"t": track_id},
    )
    from ..v1.priority import metadata_order

    order = await metadata_order()
    rank = {name: idx for idx, name in enumerate(order)}
    rows.sort(
        key=lambda r: (-int(r.get("quality_tier") or 0), rank.get(r["connector"], len(order) + 1))
    )
    return rows


async def _claim(job_id: int) -> bool:
    rows = await db.execute_returning(
        """
        UPDATE download_jobs SET status = 'running', updated_at = NOW(), attempts = attempts + 1
        WHERE id = :i AND status = 'pending'
        RETURNING id
        """,
        {"i": job_id},
    )
    return bool(rows)


class ContentMismatch(RuntimeError):
    """The staged file is a different recording than the track row claims."""


# A row can carry a junk preview duration (~30 s) while the real recording is
# minutes long; the row is the suspect there, not the file. Only rows with a
# believable duration are used to judge content identity.
_MIN_RELIABLE_DURATION_SEC = 40
_DURATION_TOLERANCE_SEC = 30


async def content_mismatch(track_id: int, measured_sec: float | None) -> str | None:
    """Return a reason when [measured] contradicts the track row, else None.

    Mislabeled upstream catalog entries (icm 1836427648 served «amy» while
    catalogued as «body double», 2026-11-18) download fine and probe fine —
    only the duration betrays them. A wrong recording must never enter the
    managed library under the right tags.
    """
    if measured_sec is None or measured_sec <= 0:
        return None
    row = await db.fetch_one("SELECT duration_sec FROM tracks WHERE id = :i", {"i": track_id})
    want = (row or {}).get("duration_sec")
    if not want or int(want) <= _MIN_RELIABLE_DURATION_SEC:
        return None
    if abs(float(measured_sec) - float(want)) > _DURATION_TOLERANCE_SEC:
        return f"content mismatch: file is {float(measured_sec):.0f}s but track is {int(want)}s"
    return None


async def _process_one(job: dict[str, Any]) -> None:
    job_id = job["id"]
    if not await _claim(job_id):
        return
    try:
        import shutil

        from . import quality
        from .library import finalize
        from .stream_service import find_cached_stream, mime_for_path

        track_id = int(job["track_id"])
        replace_local = bool(job.get("replace_local"))
        result = None

        async def verified(path: str | Path) -> None:
            """Readable audio AND the right recording — for every staged file.

            Verification used to run only for forced re-downloads and only
            checked readability, so mislabeled sources and poisoned stream-cache
            promotions entered the library under the right tags (the
            «body double»/«amy» mixup, 2026-11-18).
            """
            info = await quality.verify_file(path)
            reason = await content_mismatch(track_id, info.duration_sec)
            if reason:
                raise ContentMismatch(reason)

        async def drop_source(src: dict[str, Any]) -> None:
            # A source that serves a different recording must not stay linked.
            await db.execute(
                "UPDATE track_sources SET available = FALSE "
                "WHERE track_id = :t AND connector = :c AND external_id = :e",
                {"t": track_id, "c": src["connector"], "e": src["external_id"]},
            )

        # Promote from 200 MB LRU stream cache if already played and cached
        cached = find_cached_stream(track_id)
        if cached is not None and cached.is_file():
            dest_stage = staging_dir() / f"cached_{track_id}{cached.suffix}"
            dest_stage.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(cached, dest_stage)
            mime = mime_for_path(dest_stage)
            try:
                await verified(dest_stage)
            except Exception as exc:  # noqa: BLE001 — fall through to sources
                log.warning("stream-cache promotion rejected for track %s: %s", track_id, exc)
                dest_stage.unlink(missing_ok=True)
                cached.unlink(missing_ok=True)  # the cache entry itself is poison
            else:
                final = await finalize(track_id, dest_stage, mime)
                cached.unlink(missing_ok=True)
                log.info("library (promoted from stream cache): track %s -> %s", track_id, final)
                result = (final, mime)

        if not result:
            sources = await _sources_for(track_id)
            for src in sources:
                connector = get_connector(src["connector"])
                if connector is None:
                    continue
                log.info("downloading track %s via %s", track_id, src["connector"])
                result = await connector.download(src["external_id"], staging_dir())
                if result:
                    path, mime = result
                    try:
                        await verified(path)
                    except ContentMismatch as exc:
                        log.warning(
                            "source %s/%s serves wrong content for track %s: %s",
                            src["connector"],
                            src["external_id"],
                            track_id,
                            exc,
                        )
                        Path(path).unlink(missing_ok=True)
                        await drop_source(src)
                        result = None
                        continue
                    except Exception as exc:  # noqa: BLE001 — unreadable, try next source
                        log.warning(
                            "source %s/%s failed audio verification for track %s: %s",
                            src["connector"],
                            src["external_id"],
                            track_id,
                            exc,
                        )
                        Path(path).unlink(missing_ok=True)
                        result = None
                        continue
                    # auto-tag + atomic move into the managed tree
                    final = await finalize(track_id, path, mime)
                    log.info("library: track %s -> %s", track_id, final)
                    break
        if result and replace_local:
            try:
                await quality.quarantine_old_copies(track_id, result[0])
            except Exception:
                log.exception("quality replace cleanup failed for track %s", track_id)
        if result:
            await db.execute(
                "UPDATE download_jobs SET status = 'done', error = NULL, updated_at = NOW() WHERE id = :i",
                {"i": job_id},
            )
            try:
                from . import embed_service

                await embed_service.warm_track_for_active_widgets(track_id)
            except Exception:
                log.exception("post-download edge warm failed for track %s", track_id)
        else:
            raise RuntimeError("no source produced a file")
    except Exception as e:
        log.warning("download job %s failed: %s", job_id, e)
        attempts = job["attempts"] + 1
        status = "failed" if attempts >= MAX_ATTEMPTS else "pending"
        await db.execute(
            "UPDATE download_jobs SET status = :s, error = :e, updated_at = NOW() WHERE id = :i",
            {"s": status, "e": str(e), "i": job_id},
        )


async def _runner() -> None:
    while True:
        try:
            job = await _next_job()
        except Exception:
            log.exception("queue poll failed")
            job = None
        if job is None:
            _wake.clear()
            try:
                await asyncio.wait_for(_wake.wait(), timeout=30.0)
            except TimeoutError:
                pass
            continue
        await _process_one(job)


async def start_runner() -> None:
    global _runner_started
    if _runner_started:
        return
    _runner_started = True
    # resume interrupted jobs from a previous process
    await db.execute("UPDATE download_jobs SET status = 'pending' WHERE status = 'running'")
    asyncio.create_task(_runner())
    _wake.set()
    log.info("download runner started")
