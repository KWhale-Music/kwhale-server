"""Admin-issued playable embed widgets (Yandex-style iframe cards with Edge+Home caching)."""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal
from urllib.parse import urlparse

import httpx

from .. import auth, db
from ..config import settings
from ..errors import ProblemError
from ..v1 import images
from . import albums as albums_service
from . import artists as artists_service
from . import downloader, share_service

log = logging.getLogger("kwhale.embed")

EmbedKind = Literal["track", "album", "artist"]
EmbedTheme = Literal["auto", "dark", "light"]
EmbedLayout = Literal["auto", "compact", "full"]

_ALLOWED_KINDS: frozenset[str] = frozenset({"track", "album", "artist"})
_ALLOWED_THEMES: frozenset[str] = frozenset({"auto", "dark", "light"})
_ALLOWED_LAYOUTS: frozenset[str] = frozenset({"auto", "compact", "full"})
_ORIGIN_RE = re.compile(r"^(?:https?://)?(?:\*\.)?[a-z0-9-]+(?:\.[a-z0-9-]+)*(?::\d{1,5})?$")

_ALBUM_TRACK_CAP = 80
_ARTIST_TRACK_CAP = 25

_background_tasks: set[asyncio.Task[None]] = set()


class EmbedNotFoundError(Exception):
    """Raised when an embed widget id does not exist or is malformed."""


class EmbedInactiveError(Exception):
    """Raised when an embed widget has been disabled/revoked by admin."""


@dataclass(slots=True)
class EmbedTrackItem:
    id: int
    title: str
    artists: str
    duration_sec: int | None
    image_id: str
    local_ready: bool
    edge_warmed: bool
    edge_error: str | None


@dataclass(slots=True)
class EmbedWidgetRecord:
    id: str
    kind: EmbedKind
    entity_id: int
    title: str
    subtitle: str | None
    image_id: str
    theme: EmbedTheme
    layout: EmbedLayout
    allowed_origins: list[str]
    edge_base_url: str | None
    active: bool
    created_at: datetime
    updated_at: datetime
    tracks: list[EmbedTrackItem]


def _normalize_single_origin(raw: str) -> str:
    val = raw.strip().rstrip("/").lower()
    if not val:
        return ""
    if val in ("*", "null", "'none'", "'self'"):
        raise ProblemError(422, "validation_failed", f"Недопустимый origin: {raw}")
    if "://" in val:
        parsed = urlparse(val)
        if parsed.scheme not in ("http", "https") or parsed.path or parsed.query or parsed.fragment:
            raise ProblemError(422, "validation_failed", f"Укажите только схему и хост (без пути): {raw}")
    if not _ORIGIN_RE.fullmatch(val):
        raise ProblemError(422, "validation_failed", f"Некорректный домен/origin для iframe: {raw}")
    if ":" in val.rsplit("/", 1)[-1]:
        port_str = val.rsplit(":", 1)[-1]
        if port_str.isdigit() and not (1 <= int(port_str) <= 65535):
            raise ProblemError(422, "validation_failed", f"Некорректный порт: {raw}")
    return val


def normalize_allowed_origins(raw: list[str] | str | None) -> list[str]:
    """Parse & validate allowed iframe parent domains/origins."""
    if raw is None:
        return []
    if isinstance(raw, str):
        parts = re.split(r"[\s,;]+", raw.strip())
    else:
        parts = [str(p).strip() for p in raw]
    cleaned: list[str] = []
    seen: set[str] = set()
    for part in parts:
        p = part.strip()
        if not p:
            continue
        if p == "*":
            return []
        norm = _normalize_single_origin(p)
        if norm and norm not in seen:
            seen.add(norm)
            cleaned.append(norm)
    return cleaned[:50]


def frame_ancestors_csp(allowed_origins: list[str]) -> str:
    """Build CSP frame-ancestors value (always includes 'self' for admin preview)."""
    if not allowed_origins:
        return "frame-ancestors *"
    return "frame-ancestors 'self' " + " ".join(allowed_origins)


def normalize_edge_base_url(raw: str | None) -> str | None:
    if raw is None:
        return None
    val = raw.strip().rstrip("/")
    if not val:
        return None
    parsed = urlparse(val)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise ProblemError(422, "validation_failed", f"Некорректный URL Edge-сервера: {raw}")
    return f"{parsed.scheme}://{parsed.netloc}"


def effective_edge_base_url(widget_edge_url: str | None) -> str:
    return (widget_edge_url or settings.embed_edge_base_url or "").strip().rstrip("/")


async def _allocate_widget_id() -> str:
    for _ in range(12):
        candidate = share_service._new_share_id()
        exists = await db.fetch_one("SELECT 1 FROM embed_widgets WHERE id = :i", {"i": candidate})
        if not exists:
            return candidate
    raise RuntimeError("failed to allocate embed widget id")


async def _resolve_entity_tracks(kind: EmbedKind, entity_id: int) -> tuple[share_service.ShareEntity, list[int]]:
    entity = await share_service._load_entity(kind, entity_id)
    if entity is None:
        raise ProblemError(404, "not_found", f"{kind} #{entity_id} не найден в базе")

    if kind == "track":
        return entity, [entity_id]

    if kind == "album":
        got = await albums_service.get_album(entity_id)
        items = (got or {}).get("tracks", {}).get("items") or []
        track_ids = [int(t["id"]) for t in items if t.get("id") is not None][:_ALBUM_TRACK_CAP]
        if not track_ids:
            raise ProblemError(422, "validation_failed", "У альбома нет доступных треков для виджета")
        return entity, track_ids

    got = await artists_service.get_artist(entity_id, track_limit=_ARTIST_TRACK_CAP)
    items = (got or {}).get("top_tracks", {}).get("items") or []
    track_ids = [int(t["id"]) for t in items if t.get("id") is not None][:_ARTIST_TRACK_CAP]
    if not track_ids:
        raise ProblemError(422, "validation_failed", "У исполнителя нет доступных треков для виджета")
    return entity, track_ids


async def _pin_and_enqueue_tracks(widget_id: str, track_ids: list[int]) -> list[int]:
    """Store widget track bindings, clear pending_delete_since, and enqueue missing local files."""
    if not track_ids:
        return []
    for pos, tid in enumerate(track_ids):
        await db.execute(
            """
            INSERT INTO embed_widget_tracks (widget_id, track_id, position)
            VALUES (:w, :t, :p)
            ON CONFLICT (widget_id, track_id) DO UPDATE SET position = EXCLUDED.position
            """,
            {"w": widget_id, "t": tid, "p": pos},
        )
    placeholders = ", ".join(f":t{i}" for i in range(len(track_ids)))
    params = {f"t{i}": tid for i, tid in enumerate(track_ids)}
    await db.execute(
        f"""
        UPDATE track_sources
        SET pending_delete_since = NULL
        WHERE connector = 'local'
          AND pending_delete_since IS NOT NULL
          AND track_id IN ({placeholders})
        """,
        params,
    )
    queued: list[int] = []
    for tid in track_ids:
        job_id = await downloader.enqueue(tid, target_type="embed", target_id=None)
        if job_id is not None:
            queued.append(tid)
    return queued


async def _warm_url_on_edge(client: httpx.AsyncClient, url: str) -> None:
    async with client.stream("GET", url) as resp:
        if resp.status_code not in (200, 206):
            raise RuntimeError(f"HTTP {resp.status_code}")
        async for _chunk in resp.aiter_bytes(65536):
            pass


async def _warm_track_for_widget(
    client: httpx.AsyncClient,
    widget_id: str,
    track_id: int,
    base_url: str,
) -> bool:
    path = f"/v1/tracks/{track_id}/stream"
    exp, sig = auth.sign_media_path(path, ttl_sec=3600)
    stream_url = f"{base_url}{path}?exp={exp}&sig={sig}"
    try:
        await _warm_url_on_edge(client, stream_url)
    except Exception as exc:  # noqa: BLE001 — surfaced in admin cache status
        err = f"{type(exc).__name__}: {exc}"[:280]
        await db.execute(
            """
            UPDATE embed_widget_tracks
            SET edge_warm_error = :e
            WHERE widget_id = :w AND track_id = :t
            """,
            {"e": err, "w": widget_id, "t": track_id},
        )
        log.warning("edge warm failed for widget %s track %s: %s", widget_id, track_id, err)
        return False

    await db.execute(
        """
        UPDATE embed_widget_tracks
        SET edge_warmed_at = NOW(), edge_warm_error = NULL
        WHERE widget_id = :w AND track_id = :t
        """,
        {"w": widget_id, "t": track_id},
    )
    return True


async def warm_widget_edge(widget_id: str, *, force: bool = False) -> dict[str, int]:
    """Warm all locally downloaded tracks + cover of an active widget onto Edge cache."""
    widget = await db.fetch_one(
        "SELECT id, active, image_id, edge_base_url FROM embed_widgets WHERE id = :i",
        {"i": widget_id},
    )
    if not widget or not widget.get("active"):
        return {"warmed": 0, "failed": 0, "skipped": 0}

    base_url = effective_edge_base_url(widget.get("edge_base_url"))
    if not base_url:
        return {"warmed": 0, "failed": 0, "skipped": 1}

    rows = await db.fetch_all(
        """
        SELECT DISTINCT ewt.track_id, ewt.position, ewt.edge_warmed_at,
               EXISTS (
                   SELECT 1 FROM track_sources ts
                   WHERE ts.track_id = ewt.track_id AND ts.connector = 'local' AND ts.available = TRUE
               ) AS local_ready
        FROM embed_widget_tracks ewt
        WHERE ewt.widget_id = :w
        ORDER BY ewt.position
        """,
        {"w": widget_id},
    )
    ready_ids = [
        int(r["track_id"])
        for r in rows
        if r.get("local_ready") and (force or r.get("edge_warmed_at") is None)
    ]

    cover_rows = await db.fetch_all(
        """
        SELECT DISTINCT t.cover_url
        FROM embed_widget_tracks ewt
        JOIN tracks t ON t.id = ewt.track_id
        WHERE ewt.widget_id = :w AND COALESCE(t.cover_url, '') <> ''
        """,
        {"w": widget_id},
    )
    cover_ids: list[str] = []
    if widget.get("image_id") and widget["image_id"] != images.PLACEHOLDER_ID:
        cover_ids.append(str(widget["image_id"]))
    for cr in cover_rows:
        iid = images.image_id_for_url(cr.get("cover_url"))
        if iid != images.PLACEHOLDER_ID and iid not in cover_ids:
            cover_ids.append(iid)

    warmed = 0
    failed = 0
    sem = asyncio.Semaphore(settings.embed_warm_concurrency)
    async with httpx.AsyncClient(
        timeout=httpx.Timeout(60.0, connect=10.0),
        follow_redirects=True,
    ) as client:
        for iid in cover_ids:
            try:
                await _warm_url_on_edge(client, f"{base_url}/v1/images/{iid}?size=md")
            except Exception as exc:  # noqa: BLE001 — non-fatal cover warm
                log.debug("edge cover warm skipped for %s: %s", iid, exc)

        async def _one(tid: int) -> bool:
            async with sem:
                return await _warm_track_for_widget(client, widget_id, tid, base_url)

        results = await asyncio.gather(*(_one(tid) for tid in ready_ids))
        for ok in results:
            if ok:
                warmed += 1
            else:
                failed += 1

    return {"warmed": warmed, "failed": failed, "skipped": max(0, len(rows) - len(ready_ids))}


async def warm_track_for_active_widgets(track_id: int) -> None:
    """Called by downloader when a track finishes downloading to Home disk: warm it on Edge."""
    rows = await db.fetch_all(
        """
        SELECT ew.id, ew.edge_base_url
        FROM embed_widget_tracks ewt
        JOIN embed_widgets ew ON ew.id = ewt.widget_id
        WHERE ew.active = TRUE AND ewt.track_id = :t
        """,
        {"t": track_id},
    )
    if not rows:
        return
    async with httpx.AsyncClient(
        timeout=httpx.Timeout(60.0, connect=10.0),
        follow_redirects=True,
    ) as client:
        for r in rows:
            base_url = effective_edge_base_url(r.get("edge_base_url"))
            if not base_url:
                continue
            await _warm_track_for_widget(client, str(r["id"]), track_id, base_url)


def schedule_widget_warm(widget_id: str, *, force: bool = False) -> None:
    """Schedule background Edge warm for an embed widget without blocking HTTP responses."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return

    async def _run() -> None:
        try:
            await warm_widget_edge(widget_id, force=force)
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("background edge warm failed for widget %s", widget_id)

    task = loop.create_task(_run())
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)


async def aclose_background_tasks() -> None:
    for task in list(_background_tasks):
        if not task.done():
            task.cancel()
    _background_tasks.clear()


async def _cache_stats_map(widget_ids: list[str]) -> dict[str, dict[str, int]]:
    if not widget_ids:
        return {}
    placeholders = ", ".join(f":w{i}" for i in range(len(widget_ids)))
    params = {f"w{i}": wid for i, wid in enumerate(widget_ids)}
    rows = await db.fetch_all(
        f"""
        SELECT ewt.widget_id,
               COUNT(*) AS total,
               COUNT(*) FILTER (WHERE EXISTS (
                   SELECT 1 FROM track_sources ts
                   WHERE ts.track_id = ewt.track_id AND ts.connector = 'local' AND ts.available = TRUE
               )) AS home_ready,
               COUNT(*) FILTER (WHERE dj.status IN ('pending', 'running')) AS home_pending,
               COUNT(*) FILTER (WHERE dj.status = 'failed' AND NOT EXISTS (
                   SELECT 1 FROM track_sources ts
                   WHERE ts.track_id = ewt.track_id AND ts.connector = 'local' AND ts.available = TRUE
               )) AS home_failed,
               COUNT(*) FILTER (WHERE ewt.edge_warmed_at IS NOT NULL) AS edge_warmed,
               COUNT(*) FILTER (WHERE ewt.edge_warm_error IS NOT NULL AND ewt.edge_warmed_at IS NULL) AS edge_failed
        FROM embed_widget_tracks ewt
        LEFT JOIN download_jobs dj ON dj.track_id = ewt.track_id
        WHERE ewt.widget_id IN ({placeholders})
        GROUP BY ewt.widget_id
        """,
        params,
    )
    out: dict[str, dict[str, int]] = {}
    for r in rows:
        out[str(r["widget_id"])] = {
            "total": int(r["total"] or 0),
            "homeReady": int(r["home_ready"] or 0),
            "homePending": int(r["home_pending"] or 0),
            "homeFailed": int(r["home_failed"] or 0),
            "edgeWarmed": int(r["edge_warmed"] or 0),
            "edgeFailed": int(r["edge_failed"] or 0),
        }
    return out


def _serialize_admin_row(
    row: dict[str, Any],
    stats: dict[str, int] | None = None,
) -> dict[str, Any]:
    wid = str(row["id"])
    return {
        "id": wid,
        "kind": row["kind"],
        "entityId": str(row["entity_id"]),
        "title": row["title"],
        "subtitle": row.get("subtitle"),
        "imageId": row.get("image_id") or images.PLACEHOLDER_ID,
        "theme": row.get("theme") or "auto",
        "layout": row.get("layout") or "auto",
        "allowedOrigins": list(row.get("allowed_origins") or []),
        "edgeBaseUrl": row.get("edge_base_url") or "",
        "effectiveEdgeBaseUrl": effective_edge_base_url(row.get("edge_base_url")),
        "active": bool(row.get("active")),
        "createdBy": row.get("created_by_name"),
        "createdAt": row["created_at"].isoformat() if row.get("created_at") else None,
        "updatedAt": row["updated_at"].isoformat() if row.get("updated_at") else None,
        "cache": stats
        or {
            "total": 0,
            "homeReady": 0,
            "homePending": 0,
            "homeFailed": 0,
            "edgeWarmed": 0,
            "edgeFailed": 0,
        },
        "embedPath": f"/share/embed/{wid}",
    }


async def search_entities_for_embed(
    kind: EmbedKind, query: str, limit: int = 20
) -> list[dict[str, Any]]:
    """Search tracks, albums, or artists directly in local/aggregated DB for Admin UI."""
    if kind not in _ALLOWED_KINDS:
        raise ProblemError(400, "validation_failed", "kind must be track, album, or artist")
    q = (query or "").strip()
    exact_id = int(q) if q.isdigit() else None
    lim = max(1, min(int(limit), 50))
    like = f"%{q}%"

    if kind == "track":
        where = (
            "(t.id = :eid OR t.title ILIKE :q OR t.artists_text ILIKE :q)"
            if exact_id is not None
            else ("(t.title ILIKE :q OR t.artists_text ILIKE :q)" if q else "TRUE")
        )
        cand_order = (
            "(t.id = :eid) DESC NULLS LAST, t.last_seen_at DESC, t.id DESC"
            if exact_id is not None
            else "t.last_seen_at DESC, t.id DESC"
        )
        rows = await db.fetch_all(
            f"""
            WITH cand AS MATERIALIZED (
                SELECT t.id
                FROM tracks t
                WHERE {where}
                ORDER BY {cand_order}
                LIMIT :lim
            )
            SELECT t.id, t.title, t.artists, t.album, t.duration_sec, t.cover_url,
                   EXISTS (
                       SELECT 1 FROM track_sources s
                       WHERE s.track_id = t.id AND s.connector = 'local' AND s.available = TRUE
                   ) AS local_ready
            FROM cand c
            JOIN tracks t ON t.id = c.id
            ORDER BY (t.id = :eid) DESC NULLS LAST, local_ready DESC, t.last_seen_at DESC, t.id DESC
            """,
            {"eid": exact_id, "q": like, "lim": lim},
        )
        refs = await images.image_refs_batch([r.get("cover_url") for r in rows])
        return [
            {
                "kind": "track",
                "entityId": str(r["id"]),
                "title": str(r["title"]),
                "subtitle": ", ".join(r.get("artists") or []) or None,
                "meta": share_service.format_duration(r.get("duration_sec")) or r.get("album"),
                "imageId": refs[r["cover_url"]].id if r.get("cover_url") in refs else images.PLACEHOLDER_ID,
                "localReady": bool(r.get("local_ready")),
            }
            for r in rows
        ]

    if kind == "album":
        where = (
            "(a.id = :eid OR a.title ILIKE :q OR ar.name ILIKE :q)"
            if exact_id is not None
            else ("(a.title ILIKE :q OR ar.name ILIKE :q)" if q else "TRUE")
        )
        cand_order = (
            "(a.id = :eid) DESC NULLS LAST, a.last_seen_at DESC, a.id DESC"
            if exact_id is not None
            else "a.last_seen_at DESC, a.id DESC"
        )
        rows = await db.fetch_all(
            f"""
            WITH cand AS MATERIALIZED (
                SELECT a.id
                FROM albums a
                JOIN artists ar ON ar.id = a.artist_id
                WHERE {where}
                ORDER BY {cand_order}
                LIMIT :lim
            )
            SELECT a.id, a.title, a.year, a.track_count, a.cover_url, ar.name AS artist_name,
                   {albums_service.IN_LIBRARY_ALBUM_SQL} AS local_ready
            FROM cand c
            JOIN albums a ON a.id = c.id
            JOIN artists ar ON ar.id = a.artist_id
            ORDER BY (a.id = :eid) DESC NULLS LAST, local_ready DESC, a.year DESC NULLS LAST, lower(a.title)
            """,
            {"eid": exact_id, "q": like, "lim": lim},
        )
        refs = await images.image_refs_batch([r.get("cover_url") for r in rows])
        return [
            {
                "kind": "album",
                "entityId": str(r["id"]),
                "title": str(r["title"]),
                "subtitle": str(r["artist_name"]),
                "meta": " · ".join(
                    p
                    for p in (
                        str(r["year"]) if r.get("year") else "",
                        f"{r['track_count']} т." if r.get("track_count") else "",
                    )
                    if p
                )
                or "Альбом",
                "imageId": refs[r["cover_url"]].id if r.get("cover_url") in refs else images.PLACEHOLDER_ID,
                "localReady": bool(r.get("local_ready")),
            }
            for r in rows
        ]

    where = (
        "(ar.id = :eid OR ar.name ILIKE :q)"
        if exact_id is not None
        else ("ar.name ILIKE :q" if q else "TRUE")
    )
    cand_order = (
        "(ar.id = :eid) DESC NULLS LAST, ar.last_seen_at DESC, ar.id DESC"
        if exact_id is not None
        else "ar.last_seen_at DESC, ar.id DESC"
    )
    rows = await db.fetch_all(
        f"""
        WITH cand AS MATERIALIZED (
            SELECT ar.id
            FROM artists ar
            WHERE {where}
            ORDER BY {cand_order}
            LIMIT :lim
        )
        SELECT ar.id, ar.name, ar.cover_url, ar.genre,
               EXISTS (
                   SELECT 1 FROM track_artists ta
                   JOIN track_sources s ON s.track_id = ta.track_id
                     AND s.connector = 'local' AND s.available = TRUE
                   WHERE ta.artist_id = ar.id
               ) AS local_ready,
               (SELECT COUNT(*) FROM albums al WHERE al.artist_id = ar.id) AS album_count
        FROM cand c
        JOIN artists ar ON ar.id = c.id
        ORDER BY (ar.id = :eid) DESC NULLS LAST, local_ready DESC, album_count DESC, lower(ar.name)
        """,
        {"eid": exact_id, "q": like, "lim": lim},
    )
    refs = await images.image_refs_batch([r.get("cover_url") for r in rows])
    return [
        {
            "kind": "artist",
            "entityId": str(r["id"]),
            "title": str(r["name"]),
            "subtitle": "Исполнитель",
            "meta": f"{r['album_count']} альб." if r.get("album_count") else (r.get("genre") or "Артист"),
            "imageId": refs[r["cover_url"]].id if r.get("cover_url") in refs else images.PLACEHOLDER_ID,
            "localReady": bool(r.get("local_ready")),
        }
        for r in rows
    ]


async def create_widget(
    *,
    kind: str,
    entity_id: int,
    admin_user_id: int | None,
    title_override: str | None = None,
    theme: str = "auto",
    layout: str = "auto",
    allowed_origins: list[str] | str | None = None,
    edge_base_url: str | None = None,
) -> dict[str, Any]:
    if kind not in _ALLOWED_KINDS:
        raise ProblemError(422, "validation_failed", "kind должен быть track, album или artist")
    if theme not in _ALLOWED_THEMES:
        raise ProblemError(422, "validation_failed", "theme должен быть auto, dark или light")
    if layout not in _ALLOWED_LAYOUTS:
        raise ProblemError(422, "validation_failed", "layout должен быть auto, compact или full")

    origins = normalize_allowed_origins(allowed_origins)
    norm_edge_url = normalize_edge_base_url(edge_base_url)

    typed_kind: EmbedKind = kind  # type: ignore[assignment]
    entity, track_ids = await _resolve_entity_tracks(typed_kind, entity_id)
    wid = await _allocate_widget_id()
    final_title = (title_override or "").strip() or entity.title

    await db.execute(
        """
        INSERT INTO embed_widgets
            (id, kind, entity_id, title, subtitle, image_id, theme, layout,
             allowed_origins, edge_base_url, active, created_by, created_at, updated_at)
        VALUES
            (:id, :kind, :eid, :title, :sub, :img, :theme, :layout,
             :origins, :edge, TRUE, :uid, NOW(), NOW())
        """,
        {
            "id": wid,
            "kind": kind,
            "eid": entity_id,
            "title": final_title,
            "sub": entity.subtitle,
            "img": entity.image_id,
            "theme": theme,
            "layout": layout,
            "origins": origins,
            "edge": norm_edge_url,
            "uid": admin_user_id,
        },
    )
    await _pin_and_enqueue_tracks(wid, track_ids)
    schedule_widget_warm(wid, force=True)
    return await get_widget_admin(wid)


async def list_widgets_admin() -> list[dict[str, Any]]:
    rows = await db.fetch_all(
        """
        SELECT ew.*, u.username AS created_by_name
        FROM embed_widgets ew
        LEFT JOIN users u ON u.id = ew.created_by
        ORDER BY ew.created_at DESC
        LIMIT 200
        """
    )
    stats = await _cache_stats_map([str(r["id"]) for r in rows])
    return [_serialize_admin_row(r, stats.get(str(r["id"]))) for r in rows]


async def get_widget_admin(widget_id: str) -> dict[str, Any]:
    if not share_service._SHARE_ID_RE.fullmatch(widget_id or ""):
        raise ProblemError(404, "not_found", "Интеграция не найдена")
    row = await db.fetch_one(
        """
        SELECT ew.*, u.username AS created_by_name
        FROM embed_widgets ew
        LEFT JOIN users u ON u.id = ew.created_by
        WHERE ew.id = :i
        """,
        {"i": widget_id},
    )
    if not row:
        raise ProblemError(404, "not_found", "Интеграция не найдена")
    stats = await _cache_stats_map([widget_id])
    return _serialize_admin_row(row, stats.get(widget_id))


async def update_widget(
    widget_id: str,
    *,
    title: str | None = None,
    theme: str | None = None,
    layout: str | None = None,
    allowed_origins: list[str] | str | None = None,
    edge_base_url: str | None = None,
    active: bool | None = None,
) -> dict[str, Any]:
    existing = await db.fetch_one("SELECT * FROM embed_widgets WHERE id = :i", {"i": widget_id})
    if not existing:
        raise ProblemError(404, "not_found", "Интеграция не найдена")

    new_title = (title.strip() if title is not None else existing["title"]) or existing["title"]
    new_theme = theme if theme is not None else existing["theme"]
    if new_theme not in _ALLOWED_THEMES:
        raise ProblemError(422, "validation_failed", "theme должен быть auto, dark или light")
    new_layout = layout if layout is not None else existing["layout"]
    if new_layout not in _ALLOWED_LAYOUTS:
        raise ProblemError(422, "validation_failed", "layout должен быть auto, compact или full")
    new_origins = (
        normalize_allowed_origins(allowed_origins)
        if allowed_origins is not None
        else list(existing.get("allowed_origins") or [])
    )
    new_edge = (
        normalize_edge_base_url(edge_base_url)
        if edge_base_url is not None
        else existing.get("edge_base_url")
    )
    new_active = bool(active) if active is not None else bool(existing.get("active"))

    await db.execute(
        """
        UPDATE embed_widgets
        SET title = :title, theme = :theme, layout = :layout,
            allowed_origins = :origins, edge_base_url = :edge,
            active = :active, updated_at = NOW()
        WHERE id = :i
        """,
        {
            "i": widget_id,
            "title": new_title,
            "theme": new_theme,
            "layout": new_layout,
            "origins": new_origins,
            "edge": new_edge,
            "active": new_active,
        },
    )

    if new_active and (not existing.get("active") or (edge_base_url is not None and new_edge != existing.get("edge_base_url"))):
        tids = await db.fetch_all(
            "SELECT track_id FROM embed_widget_tracks WHERE widget_id = :w ORDER BY position",
            {"w": widget_id},
        )
        await _pin_and_enqueue_tracks(widget_id, [int(r["track_id"]) for r in tids])
        schedule_widget_warm(widget_id, force=False)

    return await get_widget_admin(widget_id)


async def trigger_widget_warm(widget_id: str, *, force: bool = True) -> dict[str, Any]:
    row = await db.fetch_one("SELECT id, kind, entity_id, active FROM embed_widgets WHERE id = :i", {"i": widget_id})
    if not row:
        raise ProblemError(404, "not_found", "Интеграция не найдена")
    if not row.get("active"):
        raise ProblemError(409, "conflict", "Нельзя прогревать отключённую интеграцию")

    typed_kind: EmbedKind = row["kind"]  # type: ignore[assignment]
    try:
        _entity, fresh_tids = await _resolve_entity_tracks(typed_kind, int(row["entity_id"]))
    except ProblemError:
        fresh_tids = []
    if fresh_tids:
        await _pin_and_enqueue_tracks(widget_id, fresh_tids)
    else:
        existing_rows = await db.fetch_all(
            "SELECT track_id FROM embed_widget_tracks WHERE widget_id = :w ORDER BY position",
            {"w": widget_id},
        )
        await _pin_and_enqueue_tracks(widget_id, [int(r["track_id"]) for r in existing_rows])
    schedule_widget_warm(widget_id, force=force)
    return await get_widget_admin(widget_id)


async def delete_widget(widget_id: str) -> bool:
    rows = await db.execute_returning(
        "DELETE FROM embed_widgets WHERE id = :i RETURNING id",
        {"i": widget_id},
    )
    return bool(rows)


async def get_public_widget(widget_id: str) -> EmbedWidgetRecord:
    if not share_service._SHARE_ID_RE.fullmatch(widget_id or ""):
        raise EmbedNotFoundError(widget_id)
    row = await db.fetch_one("SELECT * FROM embed_widgets WHERE id = :i", {"i": widget_id})
    if not row:
        raise EmbedNotFoundError(widget_id)
    if not row.get("active"):
        raise EmbedInactiveError(widget_id)

    track_rows = await db.fetch_all(
        """
        SELECT ewt.track_id, ewt.position, ewt.edge_warmed_at, ewt.edge_warm_error,
               t.title, t.artists, t.duration_sec, t.cover_url,
               EXISTS (
                   SELECT 1 FROM track_sources ts
                   WHERE ts.track_id = t.id AND ts.connector = 'local' AND ts.available = TRUE
               ) AS local_ready
        FROM embed_widget_tracks ewt
        JOIN tracks t ON t.id = ewt.track_id
        WHERE ewt.widget_id = :w
        ORDER BY ewt.position
        """,
        {"w": widget_id},
    )
    refs = await images.image_refs_batch([r.get("cover_url") for r in track_rows])
    tracks = [
        EmbedTrackItem(
            id=int(r["track_id"]),
            title=str(r["title"]),
            artists=", ".join(str(a).strip() for a in (r.get("artists") or []) if str(a).strip()),
            duration_sec=r.get("duration_sec"),
            image_id=refs[r["cover_url"]].id if r.get("cover_url") in refs else (row.get("image_id") or images.PLACEHOLDER_ID),
            local_ready=bool(r.get("local_ready")),
            edge_warmed=r.get("edge_warmed_at") is not None,
            edge_error=r.get("edge_warm_error"),
        )
        for r in track_rows
    ]
    return EmbedWidgetRecord(
        id=str(row["id"]),
        kind=row["kind"],
        entity_id=int(row["entity_id"]),
        title=str(row["title"]),
        subtitle=row.get("subtitle"),
        image_id=str(row.get("image_id") or images.PLACEHOLDER_ID),
        theme=row.get("theme") or "auto",
        layout=row.get("layout") or "auto",
        allowed_origins=list(row.get("allowed_origins") or []),
        edge_base_url=row.get("edge_base_url"),
        active=bool(row.get("active")),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        tracks=tracks,
    )


async def get_fresh_track_stream(widget_id: str, track_id: int) -> dict[str, Any]:
    """Return signed stream URL path only if widget is active and owns track_id."""
    row = await db.fetch_one(
        """
        SELECT ew.active
        FROM embed_widgets ew
        JOIN embed_widget_tracks ewt ON ewt.widget_id = ew.id
        WHERE ew.id = :w AND ewt.track_id = :t
        """,
        {"w": widget_id, "t": track_id},
    )
    if not row:
        raise EmbedNotFoundError(widget_id)
    if not row.get("active"):
        raise EmbedInactiveError(widget_id)
    path = f"/v1/tracks/{track_id}/stream"
    exp, sig = auth.sign_media_path(path, ttl_sec=3600)
    return {"streamUrl": f"{path}?exp={exp}&sig={sig}", "expiresAt": exp}
