"""Stream service: local Range-206 files, 200 MB LRU stream cache, or proxied remote streams.

Ranking: local file first, then LRU stream cache (_stream_cache/), then remote sources by quality_tier.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import time
from email.utils import formatdate
from pathlib import Path

import httpx
from fastapi import HTTPException
from fastapi.responses import FileResponse, Response, StreamingResponse

from .. import db, httpcore
from ..config import settings
from ..connectors import get_connector
from ..connectors.base import Connector, StreamInfo

log = logging.getLogger("kwhale.stream")

_REAL_ASYNC_CLIENT = httpx.AsyncClient
_proxy_clients: dict[str, httpx.AsyncClient] = {}
_warming_tracks: set[int] = set()
_warm_tasks: set[asyncio.Task[None]] = set()
_STREAM_LIMITS = httpx.Limits(max_connections=64, max_keepalive_connections=16, keepalive_expiry=30.0)


def _acquire_proxy_client(px: str | None) -> tuple[httpx.AsyncClient, bool]:
    """Return (client, should_close). Reuses a pooled keepalive client in prod; respects test mocks."""
    if httpx.AsyncClient is not _REAL_ASYNC_CLIENT:
        client = httpx.AsyncClient(
            timeout=httpx.Timeout(30.0, connect=10.0),
            follow_redirects=True,
            proxy=px,
        )
        return client, True
    key = px or "direct"
    cached_client = _proxy_clients.get(key)
    if cached_client is None or cached_client.is_closed:
        cached_client = httpx.AsyncClient(
            timeout=httpx.Timeout(30.0, connect=10.0),
            limits=_STREAM_LIMITS,
            follow_redirects=True,
            proxy=px,
        )
        _proxy_clients[key] = cached_client
    return cached_client, False


async def aclose_clients() -> None:
    """Close pooled streaming clients and cancel pending background warmups on shutdown."""
    for task in list(_warm_tasks):
        if not task.done():
            task.cancel()
    _warm_tasks.clear()
    _warming_tracks.clear()
    for client in list(_proxy_clients.values()):
        try:
            if not client.is_closed:
                await client.aclose()
        except Exception as e:  # noqa: BLE001 — shutdown path
            log.debug("stream proxy client close: %s", e)
    _proxy_clients.clear()

# Signed direct-stream URLs (Yandex get-file-info, ICM, ...) are cheap to reuse
# for a few minutes. Resolving on every play added 0.7-2.9 s before the first
# byte; the client's prefetch made it worse by hitting the same source again.
RESOLVE_TTL_SEC = 300.0
_resolve_cache: dict[tuple[str, str], tuple[float, StreamInfo]] = {}


async def _resolve_cached(connector_name: str, connector: Connector, external_id: str):
    key = (connector_name, external_id)
    cached = _resolve_cache.get(key)
    if cached is not None and cached[0] > time.monotonic():
        return cached[1]
    info = await connector.resolve_stream(external_id)
    # Only cache plain direct URLs. Assembled streams (DASH/HLS) carry
    # short-lived segment URLs and must be rebuilt on every open.
    if info is not None and info.url and not info.transform and not info.segment_urls:
        _resolve_cache[key] = (time.monotonic() + RESOLVE_TTL_SEC, info)
    return info


def _forget_resolved(connector_name: str, external_id: str) -> None:
    _resolve_cache.pop((connector_name, external_id), None)

MIME_BY_EXT: dict[str, str] = {
    ".flac": "audio/flac",
    ".m4a": "audio/mp4",
    ".mp4": "audio/mp4",
    ".aac": "audio/aac",
    ".mp3": "audio/mpeg",
    ".ogg": "audio/ogg",
    ".opus": "audio/opus",
    ".wav": "audio/wav",
    ".webm": "audio/webm",
}

EXT_BY_MIME: dict[str, str] = {
    "audio/flac": ".flac",
    "audio/alac": ".m4a",
    "audio/mp4": ".m4a",
    "audio/x-m4a": ".m4a",
    "audio/aac": ".aac",
    "audio/mpeg": ".mp3",
    "audio/ogg": ".ogg",
    "audio/opus": ".opus",
    "audio/wav": ".wav",
    "audio/webm": ".webm",
}


def mime_for_path(path: Path) -> str:
    return MIME_BY_EXT.get(path.suffix.lower(), "audio/mpeg")


def ext_for_mime(mime: str | None) -> str:
    if not mime:
        return ".mp3"
    clean = mime.split(";")[0].strip().lower()
    return EXT_BY_MIME.get(clean, ".mp3")


def stream_cache_dir() -> Path:
    d = settings.music_dir / "_stream_cache"
    d.mkdir(parents=True, exist_ok=True)
    return d


def find_cached_stream(track_id: int) -> Path | None:
    """Return cached stream file for track_id if valid and within TTL, updating LRU mtime."""
    try:
        cache_dir = stream_cache_dir()
    except OSError:
        return None
    ttl_sec = max(300.0, settings.stream_cache_ttl_hours * 3600.0)
    now = time.time()
    prefix = f"{track_id}."
    for p in cache_dir.iterdir():
        if not p.name.startswith(prefix) or p.name.endswith(".part") or not p.is_file():
            continue
        try:
            st = p.stat()
            if st.st_size <= 0:
                p.unlink(missing_ok=True)
                continue
            if now - st.st_mtime > ttl_sec:
                p.unlink(missing_ok=True)
                continue
            os.utime(p, None)  # refresh LRU timestamp
            return p
        except OSError:
            continue
    return None


def evict_stream_cache() -> dict[str, int]:
    """Enforce TTL and max size (e.g. 200 MB LRU) on _stream_cache/."""
    stats = {"evicted_files": 0, "evicted_bytes": 0}
    try:
        cache_dir = stream_cache_dir()
    except OSError:
        return stats
    max_bytes = max(10, settings.stream_cache_max_mb) * 1024 * 1024
    ttl_sec = max(300.0, settings.stream_cache_ttl_hours * 3600.0)
    now = time.time()

    entries: list[tuple[float, int, Path]] = []
    total_bytes = 0
    for p in cache_dir.iterdir():
        if not p.is_file():
            continue
        try:
            st = p.stat()
        except OSError:
            continue
        if p.name.endswith(".part"):
            if now - st.st_mtime > 600:
                p.unlink(missing_ok=True)
            continue
        if st.st_size <= 0 or (now - st.st_mtime > ttl_sec):
            try:
                p.unlink(missing_ok=True)
                stats["evicted_files"] += 1
                stats["evicted_bytes"] += st.st_size
            except OSError:
                pass
            continue
        entries.append((st.st_mtime, st.st_size, p))
        total_bytes += st.st_size

    if total_bytes > max_bytes:
        # Oldest mtime first (LRU eviction)
        entries.sort(key=lambda item: item[0])
        for _mtime, size, p in entries:
            if total_bytes <= max_bytes:
                break
            try:
                p.unlink(missing_ok=True)
                total_bytes -= size
                stats["evicted_files"] += 1
                stats["evicted_bytes"] += size
            except OSError:
                pass
    return stats


async def _load_track(track_id: int) -> tuple[dict, list[dict]]:
    track = await db.fetch_one("SELECT id, title, artists FROM tracks WHERE id = :id", {"id": track_id})
    if not track:
        raise HTTPException(404, "track not found")
    sources = await db.fetch_all(
        """
        SELECT s.id, s.connector, s.external_id, s.quality_tier, s.file_path,
               s.mime, s.available, q.error AS quality_error
        FROM track_sources s
        LEFT JOIN track_quality q ON q.local_source_id = s.id
        WHERE s.track_id = :id AND s.available
        ORDER BY CASE
                     WHEN s.connector = 'local' THEN 0
                     WHEN s.connector = 'icm' THEN 1
                     ELSE 2
                 END,
                 s.quality_tier DESC
        """,
        {"id": track_id},
    )
    if not sources:
        raise HTTPException(404, "no available sources")
    return track, sources


def _parse_single_range(range_header: str | None, total_size: int) -> tuple[int, int] | None:
    """Parse HTTP single byte Range header against total_size; raise 416 if unsatisfiable."""
    if not range_header:
        return None
    spec = range_header.strip()
    if not spec.lower().startswith("bytes="):
        return None
    first = spec[6:].split(",", 1)[0].strip()
    if "-" not in first:
        raise HTTPException(416, "invalid range", headers={"Content-Range": f"bytes */{total_size}"})
    start_s, end_s = first.split("-", 1)
    if total_size <= 0:
        raise HTTPException(416, "range not satisfiable", headers={"Content-Range": "bytes */0"})
    try:
        if start_s == "":
            suffix = int(end_s)
            if suffix <= 0:
                raise ValueError("empty suffix")
            start = max(0, total_size - suffix)
            end = total_size - 1
        elif end_s == "":
            start = int(start_s)
            end = total_size - 1
        else:
            start = int(start_s)
            end = min(int(end_s), total_size - 1)
    except ValueError as exc:
        raise HTTPException(
            416, "invalid range", headers={"Content-Range": f"bytes */{total_size}"}
        ) from exc
    if start < 0 or start >= total_size or end < start:
        raise HTTPException(
            416, "range not satisfiable", headers={"Content-Range": f"bytes */{total_size}"}
        )
    return start, end


async def _local_response(
    file_path: str,
    range_header: str | None,
    *,
    head_only: bool = False,
    cache_control: str = "private, max-age=3600",
) -> Response:
    path = Path(file_path)
    if not path.is_file():
        raise HTTPException(410, "local file missing")
    mime = mime_for_path(path)
    if head_only:
        st = path.stat()
        total = st.st_size
        bounds = _parse_single_range(range_header, total)
        etag_hash = hashlib.md5(f"{st.st_mtime}-{total}".encode(), usedforsecurity=False).hexdigest()
        headers = {
            "Accept-Ranges": "bytes",
            "Cache-Control": cache_control,
            "Last-Modified": formatdate(st.st_mtime, usegmt=True),
            "ETag": f'"{etag_hash}"',
        }
        if bounds is not None:
            start, end = bounds
            headers["Content-Range"] = f"bytes {start}-{end}/{total}"
            headers["Content-Length"] = str(end - start + 1)
            return Response(status_code=206, media_type=mime, headers=headers)
        headers["Content-Length"] = str(total)
        return Response(status_code=200, media_type=mime, headers=headers)
    # No filename= arg -> keep /stream inline (attachment stays strictly on /download).
    return FileResponse(
        path,
        media_type=mime,
        headers={"Cache-Control": cache_control},
    )


def _schedule_stream_warm(track_id: int, connector_name: str, info: StreamInfo) -> None:
    """Single-flight background warm-up into _stream_cache/ when a remote track is sliced."""
    if (
        httpx.AsyncClient is not _REAL_ASYNC_CLIENT
        or track_id in _warming_tracks
        or not info.url
        or info.transform
        or info.segment_urls
    ):
        return
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return
    if find_cached_stream(track_id) is not None:
        return
    _warming_tracks.add(track_id)
    ext = ext_for_mime(info.mime)
    url = info.url

    async def _warm() -> None:
        part_path: Path | None = None
        try:
            cdir = stream_cache_dir()
            final_path = cdir / f"{track_id}{ext}"
            if final_path.is_file() and final_path.stat().st_size > 0:
                return
            part_path = cdir / f"{track_id}_{os.getpid()}_{int(time.time() * 1000)}{ext}.part"
            px = httpcore.proxy_for(connector_name)
            client, should_close = _acquire_proxy_client(px)
            try:
                async with client.stream(
                    "GET", url, headers={"User-Agent": httpcore.CHROME_UA}
                ) as resp:
                    if resp.status_code != 200:
                        return
                    expected = resp.headers.get("content-length")
                    written = 0
                    with open(part_path, "wb") as f:
                        async for chunk in resp.aiter_bytes(settings.stream_chunk_size):
                            f.write(chunk)
                            written += len(chunk)
                if written > 0 and (not expected or not expected.isdigit() or written == int(expected)):
                    os.replace(part_path, final_path)
                    evict_stream_cache()
                    part_path = None
            finally:
                if should_close:
                    await client.aclose()
        except Exception as exc:  # noqa: BLE001 — opportunistic cache warmup
            log.debug("background stream warm failed for %s: %s", track_id, exc)
        finally:
            if part_path is not None:
                part_path.unlink(missing_ok=True)
            _warming_tracks.discard(track_id)

    task = loop.create_task(_warm())
    _warm_tasks.add(task)
    task.add_done_callback(_warm_tasks.discard)


async def stream_response(
    track_id: int,
    range_header: str | None,
    *,
    head_only: bool = False,
    cache_control: str = "private, max-age=3600",
) -> Response:
    _, sources = await _load_track(track_id)

    # 1) permanent local source in Library. A local file can exist yet be
    # corrupt (quality scan found 257 truncated m4a files); those must fall
    # through to the stream cache/remotes instead of serving garbage with 206.
    for local in (
        s for s in sources if s["connector"] == "local" and s["file_path"]
    ):
        if local.get("quality_error"):
            log.warning(
                "skip corrupt local track %s (%s): %s",
                track_id,
                local["file_path"],
                local["quality_error"],
            )
            continue
        try:
            return await _local_response(
                local["file_path"],
                range_header,
                head_only=head_only,
                cache_control=cache_control,
            )
        except HTTPException as e:
            if e.status_code != 410:
                raise
            # file gone: mark source unavailable, try the next local copy
            await db.execute(
                "UPDATE track_sources SET available = FALSE WHERE id = :sid",
                {"sid": local["id"]},
            )

    # 2) 200 MB LRU stream cache (instant Range/206 seek on recently played tracks)
    cached = find_cached_stream(track_id)
    if cached is not None:
        return await _local_response(
            str(cached),
            range_header,
            head_only=head_only,
            cache_control=cache_control,
        )

    # 3) remote sources by rank
    for source in sources:
        if source["connector"] == "local":
            continue
        connector = get_connector(source["connector"])
        if connector is None:
            continue
        try:
            info = await _resolve_cached(
                source["connector"], connector, source["external_id"]
            )
        except Exception as e:
            log.warning("resolve %s/%s failed: %s", source["connector"], source["external_id"], e)
            _forget_resolved(source["connector"], source["external_id"])
            continue
        if info is None or not info.url:
            await db.execute(
                "UPDATE track_sources SET available = FALSE WHERE id = :sid",
                {"sid": source["id"]},
            )
            continue
        try:
            if connector.needs_assembly(info):
                resp = await _assembled_response(
                    connector,
                    info,
                    range_header,
                    track_id=track_id,
                    connector_name=source["connector"],
                    head_only=head_only,
                )
                resp.headers["Cache-Control"] = cache_control
                return resp
            if head_only:
                resp = await _proxy_head_response(
                    info,
                    range_header,
                    connector_name=source["connector"],
                )
            else:
                resp = await _proxy_response(
                    info,
                    range_header,
                    track_id=track_id,
                    connector_name=source["connector"],
                    connector=connector,
                )
                if range_header:
                    _schedule_stream_warm(track_id, source["connector"], info)
            resp.headers["Cache-Control"] = cache_control
            return resp
        except HTTPException as e:
            if e.status_code not in (404, 502, 503):
                raise
            log.warning(
                "source %s/%s failed (%s), trying next",
                source["connector"],
                source["external_id"],
                e.detail,
            )
            _forget_resolved(source["connector"], source["external_id"])
            continue
        except Exception as e:  # noqa: BLE001 — one dead source must not kill the track
            log.warning(
                "source %s/%s failed (%s), trying next",
                source["connector"],
                source["external_id"],
                e,
            )
            _forget_resolved(source["connector"], source["external_id"])
            continue

    raise HTTPException(404, "no playable source")


async def _assembled_response(
    connector: Connector,
    info: StreamInfo,
    range_header: str | None = None,
    *,
    track_id: int | None = None,
    connector_name: str = "providers",
    head_only: bool = False,
) -> Response:
    """Cache-aware assembly of a segmented stream — the connector owns the protocol.

    The service only handles temp-file/LRU-cache bookkeeping; how a segmented
    stream (DASH/HLS concat, vk_hls, ...) becomes a single media file is the
    connector's business (Connector.assemble_stream).
    """
    import tempfile

    from starlette.background import BackgroundTask

    ext = ext_for_mime(info.mime or "audio/mp4")
    use_cache = track_id is not None
    if use_cache:
        cache_dir = stream_cache_dir()
        # Unique part per assembly: two concurrent requests for the same track
        # used to share "{id}.part", and one os.replace() made the other fail
        # with FileNotFoundError (the 40467 assembly incident).
        part_path = cache_dir / f"{track_id}_{os.getpid()}_{int(time.time() * 1000)}{ext}.part"
        final_path = cache_dir / f"{track_id}{ext}"
    else:
        with tempfile.NamedTemporaryFile(prefix="kwhale-dash-", suffix=ext, delete=False) as tf:
            part_path = Path(tf.name)
            final_path = part_path

    try:
        await connector.assemble_stream(info, part_path)
        if use_cache:
            os.replace(part_path, final_path)
            evict_stream_cache()
    except HTTPException:
        part_path.unlink(missing_ok=True)
        raise
    except Exception as e:  # noqa: BLE001
        part_path.unlink(missing_ok=True)
        log.warning("stream assemble failed (%s): %s", connector_name, e)
        raise HTTPException(502, "stream assembly failed") from e

    if head_only:
        resp = await _local_response(str(final_path), range_header, head_only=True)
        if not use_cache:
            final_path.unlink(missing_ok=True)
        return resp

    bg = None
    if not use_cache:
        def _cleanup() -> None:
            final_path.unlink(missing_ok=True)
        bg = BackgroundTask(_cleanup)

    return FileResponse(
        final_path,
        media_type=info.mime or mime_for_path(final_path),
        background=bg,
        headers={"Cache-Control": "private, max-age=3600"},
    )


# upstream response headers relayed to the client verbatim (seek support)
_RELAY_HEADERS = ("content-range", "content-length", "accept-ranges")


def _range_start(content_range: str | None) -> int:
    """'bytes 100-199/4000' -> 100 (absolute offset for position-aware transforms)."""
    if not content_range:
        return 0
    try:
        spec = content_range.split()[1].split("-")[0]
        return int(spec)
    except (IndexError, ValueError):
        return 0


def _content_range_parts(content_range: str | None) -> tuple[int | None, int | None, int | None]:
    """'bytes 100-199/4000' -> (100, 199, 4000). Unknown/absent -> (None, None, None)."""
    if not content_range:
        return None, None, None
    try:
        range_part, total_part = content_range.split()[1].split("/")
        start_s, end_s = range_part.split("-")
        total = None if total_part == "*" else int(total_part)
        return int(start_s), int(end_s), total
    except (IndexError, ValueError):
        return None, None, None


def _safe_url(url: str) -> str:
    """URL without query/fragment — signed tokens must never reach the logs."""
    return url.split("?", 1)[0].split("#", 1)[0]


def _is_full_response(resp: httpx.Response, start: int) -> bool:
    """True only when the upstream body is the ENTIRE file (safe to cache complete).

    A partial 206 (e.g. Range: bytes=0-1023 -> Content-Range: bytes 0-1023/5000000)
    must never be promoted to the cached "full" file — that would poison the cache
    for every later listener of the track.
    """
    if resp.status_code == 200:
        return True
    if resp.status_code == 206 and start == 0:
        _s, end, total = _content_range_parts(resp.headers.get("content-range"))
        return end is not None and total is not None and end + 1 == total
    return False


async def _proxy_head_response(
    info: StreamInfo,
    range_header: str | None = None,
    *,
    connector_name: str = "providers",
) -> Response:
    """Resolve upstream headers for HEAD without transferring the audio payload."""
    assert info.url is not None, "proxy needs remote url"
    url: str = info.url
    upstream_headers: dict[str, str] = {"User-Agent": httpcore.CHROME_UA}
    if range_header:
        upstream_headers["Range"] = range_header
    px = httpcore.proxy_for(connector_name)
    client, should_close = _acquire_proxy_client(px)
    try:
        resp = await client.head(url, headers=upstream_headers)
        if resp.status_code in (200, 206):
            relay = {k: v for k, v in resp.headers.items() if k.lower() in _RELAY_HEADERS}
            media_type = info.mime or resp.headers.get("content-type", "audio/mpeg").split(";")[0]
            return Response(status_code=resp.status_code, media_type=media_type, headers=relay)
        if resp.status_code not in (405, 501):
            log.warning("upstream HEAD %s: HTTP %s", _safe_url(url), resp.status_code)
            raise HTTPException(502, "upstream stream failed")
        async with client.stream("GET", url, headers=upstream_headers) as get_resp:
            if get_resp.status_code not in (200, 206):
                log.warning("upstream GET-HEAD %s: HTTP %s", _safe_url(url), get_resp.status_code)
                raise HTTPException(502, "upstream stream failed")
            relay = {k: v for k, v in get_resp.headers.items() if k.lower() in _RELAY_HEADERS}
            media_type = info.mime or get_resp.headers.get("content-type", "audio/mpeg").split(";")[0]
            return Response(status_code=get_resp.status_code, media_type=media_type, headers=relay)
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001
        log.warning("upstream HEAD %s failed: %s", _safe_url(url), e)
        raise HTTPException(502, "upstream stream failed") from e
    finally:
        if should_close:
            await client.aclose()


async def _proxy_response(
    info: StreamInfo,
    range_header: str | None = None,
    *,
    track_id: int | None = None,
    connector_name: str = "providers",
    connector: Connector | None = None,
) -> Response:
    """Chunked passthrough proxy with modular proxy support and LRU stream caching.

    Forwards the client's Range header upstream, opens the upstream stream
    eagerly (to read status + headers) and relays 200/206 +
    Content-Range/Content-Length/Accept-Ranges/Content-Type so players can
    seek remote tracks and know durations. When streaming from byte 0, tees
    bytes into _stream_cache/ (up to 200 MB LRU) so subsequent seeks/replays
    are served straight from local disk.
    """
    assert info.url is not None, "proxy needs remote url"
    url: str = info.url
    upstream_headers: dict[str, str] = {"User-Agent": httpcore.CHROME_UA}
    if range_header:
        upstream_headers["Range"] = range_header
    px = httpcore.proxy_for(connector_name)
    client, should_close = _acquire_proxy_client(px)

    # eager open: read status + relay headers before constructing the response
    resp_ctx = client.stream("GET", url, headers=upstream_headers)
    try:
        resp = await resp_ctx.__aenter__()
    except Exception as e:  # noqa: BLE001
        if should_close:
            await client.aclose()
        log.warning("upstream connect %s failed: %s", _safe_url(url), e)
        raise HTTPException(502, "upstream stream failed") from e

    if resp.status_code not in (200, 206):
        log.warning("upstream %s: HTTP %s", _safe_url(url), resp.status_code)
        await resp_ctx.__aexit__(None, None, None)
        if should_close:
            await client.aclose()
        raise HTTPException(502, "upstream stream failed")

    relay = {k: v for k, v in resp.headers.items() if k.lower() in _RELAY_HEADERS}
    media_type = info.mime or resp.headers.get("content-type", "audio/mpeg").split(";")[0]
    start = _range_start(resp.headers.get("content-range"))
    expected_len: int | None = None
    if resp.headers.get("content-length", "").isdigit():
        expected_len = int(resp.headers["content-length"])
    _rs, _re, content_total = _content_range_parts(resp.headers.get("content-range"))
    expected_total = content_total if content_total is not None else expected_len

    # Tee into _stream_cache/ only for COMPLETE upstream bodies. A partial 206
    # (e.g. bytes=0-1023) must not be promoted to the full cached file.
    can_cache = track_id is not None and start == 0 and _is_full_response(resp, start)
    ext = ext_for_mime(media_type)
    part_path: Path | None = None
    final_path: Path | None = None
    if can_cache:
        try:
            cdir = stream_cache_dir()
            part_path = cdir / f"{track_id}_{os.getpid()}_{int(time.time() * 1000)}{ext}.part"
            final_path = cdir / f"{track_id}{ext}"
        except OSError:
            part_path = None
            final_path = None

    dec = None if connector is None else connector.stream_decoder(info, url)

    # Read the first chunk *before* committing the response. A dead upstream
    # often fails here (Yandex returns headers, then closes mid-body), and in
    # that case stream_response can still fall back to the next source. Once
    # the first byte is out we can only truncate and let the player retry.
    iterator = resp.aiter_bytes(settings.stream_chunk_size)
    first_chunk: bytes | None = None
    try:
        first_chunk = await iterator.__anext__()
    except StopAsyncIteration:
        raise HTTPException(502, "upstream stream empty") from None
    except Exception as e:  # noqa: BLE001
        log.warning("upstream read %s failed before first byte: %s", _safe_url(url), e)
        raise HTTPException(502, "upstream stream failed") from e
    finally:
        # Close on error/cancellation; on success gen_stream owns the context.
        if first_chunk is None:
            await resp_ctx.__aexit__(None, None, None)
            if should_close:
                await client.aclose()

    async def gen_stream():
        offset = start
        written = 0
        cache_f = None
        if part_path is not None:
            try:
                cache_f = open(part_path, "wb")  # noqa: SIM115
            except OSError:
                cache_f = None
        completed = False
        try:
            chunk = first_chunk
            while True:
                out_chunk = dec.process(chunk, offset) if dec is not None else chunk
                offset += len(chunk)
                if cache_f is not None:
                    try:
                        cache_f.write(out_chunk)
                        written += len(out_chunk)
                    except OSError:
                        cache_f.close()
                        cache_f = None
                yield out_chunk
                try:
                    chunk = await iterator.__anext__()
                except StopAsyncIteration:
                    break
                except Exception as e:  # noqa: BLE001
                    # Response headers are already sent; log and close cleanly.
                    # The client sees a short body and retries once (0.1.10).
                    log.warning(
                        "upstream read %s failed after %d bytes: %s",
                        _safe_url(url),
                        written,
                        e,
                    )
                    break
            size_ok = (expected_len is None or written == expected_len) and (
                expected_total is None or written == expected_total
            )
            if size_ok:
                completed = written > 0
        finally:
            if cache_f is not None:
                try:
                    cache_f.close()
                except OSError:
                    pass
            if part_path is not None:
                if completed and final_path is not None:
                    try:
                        os.replace(part_path, final_path)
                        evict_stream_cache()
                    except OSError:
                        part_path.unlink(missing_ok=True)
                else:
                    part_path.unlink(missing_ok=True)
            await resp_ctx.__aexit__(None, None, None)
            if should_close:
                await client.aclose()

    relay["Cache-Control"] = "private, max-age=3600"
    return StreamingResponse(
        gen_stream(),
        status_code=resp.status_code,
        media_type=media_type,
        headers=relay,
    )


def _safe_filename(track: dict, suffix: str) -> str:
    import re

    artists = ", ".join(track.get("artists") or []) or "Unknown artist"
    raw = f"{artists} - {track.get('title') or 'track'}{suffix}"
    cleaned = re.sub(r'[\\/:*?"<>|\x00-\x1f]', "_", raw).strip()
    return cleaned[:180] or f"track-{track.get('id', 'audio')}.mp3"


def _content_disposition(filename: str, fallback_prefix: str = "track") -> str:
    import urllib.parse
    ascii_clean = filename.encode("ascii", "ignore").decode().strip()
    if not ascii_clean or ascii_clean.startswith("."):
        suffix = Path(filename).suffix or ".mp3"
        ascii_name = f"{fallback_prefix}{suffix}"
    else:
        ascii_name = ascii_clean
    encoded = urllib.parse.quote(filename, safe="")
    return f'attachment; filename="{ascii_name}"; filename*=UTF-8\'\'{encoded}'


async def download_response(track_id: int) -> Response:
    """Same-origin attachment: local file or proxied remote source."""
    track, _sources = await _load_track(track_id)
    resp = await stream_response(track_id, None)
    suffix = ".mp3"
    path = getattr(resp, "path", None)
    if path:
        suffix = Path(path).suffix or suffix
    filename = _safe_filename(track, suffix)
    resp.headers["Content-Disposition"] = _content_disposition(filename, f"track-{track_id}")
    resp.headers["Cache-Control"] = "private, max-age=0"
    return resp
