"""Maintenance loop: daily ICM keepalive + GC + fsck + dropbox watcher.

Dropbox watch: files in data/music/_dropbox/ are picked up by the manual
pipeline (fingerprint -> tag -> tree) once their size is stable for
DROPBOX_STABLE_SECONDS.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from ..config import settings
from ..connectors import get_connectors
from .gc import purge_search_cache, run_gc
from .library import fsck

log = logging.getLogger("kwhale.maintenance")

_MAINTENANCE_PERIOD = 24 * 3600  # seconds
_DROPBOX_POLL = 30  # seconds


async def _set_status(name: str, status: str, message: str = "") -> None:
    from .. import db

    try:
        await db.execute(
            """
            INSERT INTO service_status (name, status, message, updated_at)
            VALUES (:n, :s, :m, NOW())
            ON CONFLICT (name) DO UPDATE
            SET status = EXCLUDED.status, message = EXCLUDED.message, updated_at = NOW()
            """,
            {"n": name, "s": status, "m": message},
        )
    except Exception:
        log.exception("service status update failed")


async def _icm_keepalive_once() -> bool:
    connectors = get_connectors()
    icm = connectors.get("icm")
    if icm is None or not await icm.is_available():
        await _set_status("icm_keepalive", "skipped", "ICM is not configured")
        return True
    try:
        ok = await icm.keepalive()
        await _set_status("icm_keepalive", "ok" if ok else "failed", "" if ok else "keepalive returned false")
        log.info("icm keepalive: %s", "ok" if ok else "FAILED")
        return ok
    except Exception as e:
        await _set_status("icm_keepalive", "failed", str(e))
        log.exception("icm keepalive error")
        return False


async def _icm_loop() -> None:
    await asyncio.sleep(60)
    while True:
        ok = await _icm_keepalive_once()
        if not ok:
            await asyncio.sleep(300)
            await _icm_keepalive_once()
        await asyncio.sleep(max(1, settings.icm_keepalive_hours) * 3600)


async def _cycle() -> None:
    try:
        await run_gc()
    except Exception:
        log.exception("gc error")

    try:
        stats = await fsck()
        if any(stats.values()):
            log.info("fsck done: %s", stats)
    except Exception:
        log.exception("fsck error")

    try:
        from . import quality

        if not await quality.start_scan(limit=1000):
            log.info("quality scan already running")
    except Exception:
        log.exception("quality scan error")

    try:
        from .rec.stats import refresh_stats

        await refresh_stats()
    except Exception:
        log.exception("recommendation stats refresh error")

    try:
        from .weekly_releases import run as run_weekly_releases

        await run_weekly_releases()
    except Exception:
        log.exception("weekly releases error")


def _is_stable(path: Path, seen: dict[Path, tuple[int, float]], min_age: float) -> bool:
    """File is stable when size unchanged between polls AND mtime older than min_age."""
    import time

    try:
        stat = path.stat()
    except OSError:
        return False
    prev = seen.get(path)
    seen[path] = (stat.st_size, stat.st_mtime)
    if prev is not None and prev[0] == stat.st_size and time.time() - stat.st_mtime > min_age:
        return True
    return False


async def _dropbox_loop() -> None:
    """Watch _dropbox/ for manual files (poll, size-stability debounce)."""
    from .library import dropbox_dir
    from .manual import is_audio, process_manual_file

    seen: dict[Path, tuple[int, float]] = {}
    while True:
        await asyncio.sleep(_DROPBOX_POLL)
        try:
            root = dropbox_dir()
            files = [p for p in root.rglob("*") if p.is_file() and is_audio(p)]
            # purge stale entries from the seen-map
            live = set(files)
            for p in list(seen):
                if p not in live:
                    seen.pop(p, None)
            for p in files:
                if _is_stable(p, seen, min_age=settings.dropbox_stable_seconds):
                    seen.pop(p, None)
                    log.info("dropbox: processing %s", p.name)
                    try:
                        await process_manual_file(p)
                    except Exception as e:
                        log.exception("dropbox processing %s failed: %s", p.name, e)
        except Exception:
            log.exception("dropbox watcher error")


async def _cache_cleanup_loop() -> None:
    """Search cache is cheap to clean; do not wait for the 24h GC cycle."""
    while True:
        try:
            removed = await purge_search_cache()
            if removed:
                log.info("search cache: removed %s expired rows", removed)
        except Exception:
            log.exception("search cache cleanup failed")
        await asyncio.sleep(3600)


async def start() -> None:
    asyncio.create_task(_loop())
    asyncio.create_task(_cache_cleanup_loop())
    asyncio.create_task(_dropbox_loop())
    asyncio.create_task(_icm_loop())
    log.info(
        "maintenance started (icm keepalive every %sh, dropbox poll %ss)",
        settings.icm_keepalive_hours,
        _DROPBOX_POLL,
    )


async def _loop() -> None:
    # first cycle after short delay to let the app boot
    await asyncio.sleep(60)
    while True:
        await _cycle()
        await asyncio.sleep(_MAINTENANCE_PERIOD)
