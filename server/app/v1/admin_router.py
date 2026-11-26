"""Internal admin API (Bearer admin, not part of the public KWhale contract)."""

from __future__ import annotations

import logging
from typing import Any, Literal

from fastapi import APIRouter, Depends, Query, Request, Response
from pydantic import BaseModel

from ..auth import (
    create_invite,
    list_invites,
    list_users,
    require_admin,
    revoke_invite,
    set_role,
)
from ..errors import ProblemError
from ..services import quality
from ..services import source_accounts as accounts
from ..services.rec import analytics as rec_analytics
from ..services.source_accounts import SOURCE_SPECS

log = logging.getLogger("kwhale.admin")

router = APIRouter(prefix="/admin/api", tags=["admin"])


class ConnectRequest(BaseModel):
    fields: dict[str, str] = {}


class EnabledRequest(BaseModel):
    enabled: bool


class InviteRequest(BaseModel):
    days: int = 7


class RoleRequest(BaseModel):
    role: Literal["user", "admin"]


class PriorityRequest(BaseModel):
    order: list[str]


class ProxySettingsRequest(BaseModel):
    enabled: bool = True
    defaultProxy: str = ""
    overrides: dict[str, str] = {}


class QualityIdsRequest(BaseModel):
    trackIds: list[int] = []


class QualityScanRequest(BaseModel):
    trackIds: list[int] | None = None
    all: bool = False
    force: bool = False
    limit: int | None = None


class EmbedCreateRequest(BaseModel):
    kind: Literal["track", "album", "artist"]
    entityId: int
    title: str | None = None
    theme: Literal["auto", "dark", "light"] = "auto"
    layout: Literal["auto", "compact", "full"] = "auto"
    allowedOrigins: list[str] | str = []
    edgeBaseUrl: str | None = None


class EmbedUpdateRequest(BaseModel):
    title: str | None = None
    theme: Literal["auto", "dark", "light"] | None = None
    layout: Literal["auto", "compact", "full"] | None = None
    allowedOrigins: list[str] | str | None = None
    edgeBaseUrl: str | None = None
    active: bool | None = None


class EmbedWarmRequest(BaseModel):
    force: bool = True


@router.get("/me")
async def me(admin: dict = Depends(require_admin)) -> dict[str, Any]:
    return {
        "id": str(admin["id"]),
        "displayName": admin.get("display_name") or admin.get("username"),
        "email": admin.get("email"),
        "role": admin.get("role"),
    }


@router.get("/sources")
async def list_sources(admin: dict = Depends(require_admin)) -> dict[str, Any]:
    return {"items": await accounts.admin_view()}


@router.post("/sources/{name}/connect")
async def connect_source(
    name: str, body: ConnectRequest, admin: dict = Depends(require_admin)
) -> dict[str, Any]:
    spec = SOURCE_SPECS.get(name)
    if spec is None:
        raise ProblemError(404, "not_found", f"unknown source {name}")
    if spec.get("auth") == "env":
        raise ProblemError(400, "validation_failed", f"{name} is configured through .env")
    fields = {k: (v or "").strip() for k, v in body.fields.items() if (v or "").strip()}
    required = list(spec.get("fields") or [])
    alt = list(spec.get("alt_fields") or [])
    opt = list(spec.get("optional_fields") or [])
    if required and not any(f in fields for f in required) and not (alt and any(f in fields for f in alt)):
        raise ProblemError(422, "validation_failed", f"missing fields: {'/'.join(required)}")
    if not required and opt and not any(f in fields for f in opt):
        raise ProblemError(422, "validation_failed", f"missing fields: {'/'.join(opt)}")

    from ..connectors import get_connector

    conn = get_connector(name)
    if conn is None or not hasattr(conn, "connect"):
        raise ProblemError(501, "not_implemented", f"connector {name} has no connect flow")
    state = await conn.connect(fields)
    if not state.ok:
        raise ProblemError(422, "validation_failed", state.message or "connect failed")
    payload = {**fields, **state.payload_updates}
    await accounts.save_account(name, payload, profile=state.profile, status=state.status)
    log.info("source %s connected (%s)", name, state.profile.get("name", "?"))
    return {"ok": True, "status": state.status, "profile": state.profile}


@router.post("/sources/{name}/check")
async def check_source(name: str, admin: dict = Depends(require_admin)) -> dict[str, Any]:
    from ..connectors import get_connector

    conn = get_connector(name)
    if conn is None or not hasattr(conn, "check_account"):
        raise ProblemError(501, "not_implemented", f"connector {name} has no account check")
    state = await conn.check_account()
    if state.payload_updates:
        await accounts.update_payload(name, state.payload_updates)
    await accounts.set_profile(name, profile=state.profile, status=state.status)
    return {"ok": state.ok, "status": state.status, "profile": state.profile, "message": state.message}


@router.post("/sources/{name}/enabled")
async def set_source_enabled(
    name: str, body: EnabledRequest, admin: dict = Depends(require_admin)
) -> dict[str, Any]:
    if name not in SOURCE_SPECS:
        raise ProblemError(404, "not_found", f"unknown source {name}")
    await accounts.set_enabled(name, body.enabled)
    return {"ok": True, "name": name, "enabled": body.enabled}


@router.delete("/sources/{name}")
async def disconnect_source(name: str, admin: dict = Depends(require_admin)) -> dict[str, Any]:
    if name not in SOURCE_SPECS:
        raise ProblemError(404, "not_found", f"unknown source {name}")
    await accounts.disconnect(name)
    return {"ok": True, "name": name}


@router.get("/metadata-priority")
async def get_metadata_priority(admin: dict = Depends(require_admin)) -> dict[str, Any]:
    from .priority import priority_rows

    return {"items": await priority_rows()}


@router.put("/metadata-priority")
async def set_metadata_priority(
    body: PriorityRequest, admin: dict = Depends(require_admin)
) -> dict[str, Any]:
    from .priority import set_metadata_order

    try:
        order = await set_metadata_order(body.order)
    except ValueError as e:
        raise ProblemError(400, "validation_failed", str(e)) from e
    return {"items": [{"connector": n, "priority": (i + 1) * 10} for i, n in enumerate(order)]}


@router.get("/proxy")
async def get_proxy(admin: dict = Depends(require_admin)) -> dict[str, Any]:
    from .. import httpcore

    return httpcore.get_proxy_settings()


@router.put("/proxy")
async def update_proxy(
    body: ProxySettingsRequest, admin: dict = Depends(require_admin)
) -> dict[str, Any]:
    from .. import httpcore

    return await httpcore.update_proxy_settings(
        enabled=body.enabled,
        default_proxy=body.defaultProxy,
        overrides=body.overrides,
    )


@router.get("/invites")
async def invites(admin: dict = Depends(require_admin)) -> dict[str, Any]:
    return {"items": await list_invites()}


@router.post("/invites", status_code=201)
async def create_invite_endpoint(
    body: InviteRequest, admin: dict = Depends(require_admin)
) -> dict[str, Any]:
    return await create_invite(admin["id"], body.days)


@router.delete("/invites/{invite_id}", status_code=204)
async def revoke_invite_endpoint(invite_id: int, admin: dict = Depends(require_admin)) -> Response:
    if not await revoke_invite(invite_id):
        raise ProblemError(404, "not_found", "Invite not found or already used")
    return Response(status_code=204)


@router.get("/users")
async def users(admin: dict = Depends(require_admin)) -> dict[str, Any]:
    return {"items": await list_users()}


@router.patch("/users/{user_id}")
async def update_user_role(
    user_id: int, body: RoleRequest, admin: dict = Depends(require_admin)
) -> dict[str, Any]:
    if not await set_role(user_id, body.role):
        raise ProblemError(404, "not_found", "User not found")
    return {"ok": True, "id": str(user_id), "role": body.role}


@router.get("/status")
async def status(admin: dict = Depends(require_admin)) -> dict[str, Any]:
    row = await _service_status("icm_keepalive")
    return {"icmKeepalive": row}


# ── Library audio quality ───────────────────────────────────────────────────


def _quality_ids(body: QualityIdsRequest) -> list[int]:
    return list(dict.fromkeys(int(v) for v in body.trackIds))[:500]


@router.get("/quality/overview")
async def quality_overview(
    days: int = Query(30, ge=0, le=3650),
    admin: dict = Depends(require_admin),
) -> dict[str, Any]:
    return await quality.overview(days)


@router.get("/quality/tracks")
async def quality_tracks(
    days: int = Query(30, ge=0, le=3650),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    maxBitrate: int | None = Query(None, ge=0, le=10000),
    minBitrate: int | None = Query(None, ge=0, le=10000),
    lossless: bool | None = Query(None),
    codec: str | None = Query(None, max_length=32),
    tier: int | None = Query(None, ge=0, le=6),
    q: str | None = Query(None, max_length=200),
    played: bool | None = Query(None),
    downloaded: bool | None = Query(None),
    sort: str = Query("worst"),
    admin: dict = Depends(require_admin),
) -> dict[str, Any]:
    return await quality.list_tracks(
        days=days,
        limit=limit,
        offset=offset,
        max_bitrate=maxBitrate,
        min_bitrate=minBitrate,
        lossless=lossless,
        codec=codec,
        tier=tier,
        query=q,
        played=played,
        downloaded=downloaded,
        sort=sort,
    )


@router.get("/quality/scan")
async def quality_scan_status(admin: dict = Depends(require_admin)) -> dict[str, Any]:
    return quality.scan_status()


@router.post("/quality/rescan")
async def quality_rescan(
    body: QualityScanRequest,
    admin: dict = Depends(require_admin),
) -> dict[str, Any]:
    ids = list(dict.fromkeys(int(v) for v in (body.trackIds or [])))[:500]
    if ids:
        result = await quality.scan(
            limit=body.limit,
            force=body.force,
            track_ids=ids,
        )
        return {"mode": "selected", **result}
    started = await quality.start_scan(force=body.force, limit=body.limit)
    return {"mode": "all", "started": started, **quality.scan_status()}


@router.post("/quality/redownload")
async def quality_redownload(
    body: QualityIdsRequest,
    admin: dict = Depends(require_admin),
) -> dict[str, Any]:
    return await quality.redownload(_quality_ids(body))


@router.post("/quality/delete")
async def quality_delete(
    body: QualityIdsRequest,
    admin: dict = Depends(require_admin),
) -> dict[str, Any]:
    return await quality.delete_local(_quality_ids(body))


# ── Recommendations ────────────────────────────────────────────────────────


@router.get("/recommendations/overview")
async def recommendations_overview(
    admin: dict = Depends(require_admin),
) -> dict[str, Any]:
    return await rec_analytics.overview()


# ── Censorship review ──────────────────────────────────────────────────────


@router.get("/censorship")
async def list_censorship_reports(
    status: str = "pending",
    admin: dict = Depends(require_admin),
) -> dict[str, Any]:
    """Tracks users sent for censorship review («Отправить на цензуру»)."""
    from .. import db

    rows = await db.fetch_all(
        """
        SELECT cr.id, cr.track_id, cr.reason, cr.comment, cr.position_ms,
               cr.status, cr.created_at, u.username,
               t.title, t.artists
        FROM censorship_reports cr
        JOIN users u ON u.id = cr.user_id
        JOIN tracks t ON t.id = cr.track_id
        WHERE (:s = 'all' OR cr.status = :s)
        ORDER BY cr.created_at DESC
        LIMIT 200
        """,
        {"s": status},
    )
    return {
        "items": [
            {
                "id": r["id"],
                "trackId": r["track_id"],
                "title": r["title"],
                "artists": r["artists"],
                "reason": r["reason"],
                "comment": r["comment"],
                "positionMs": r["position_ms"],
                "status": r["status"],
                "user": r["username"],
                "createdAt": r["created_at"].isoformat() if r.get("created_at") else None,
            }
            for r in rows
        ]
    }


class CensorshipResolveRequest(BaseModel):
    status: str = "resolved"


@router.post("/censorship/{report_id}/resolve")
async def resolve_censorship_report(
    report_id: int,
    body: CensorshipResolveRequest,
    admin: dict = Depends(require_admin),
) -> dict[str, Any]:
    from .. import db

    allowed = {"resolved", "rejected", "pending"}
    new_status = body.status if body.status in allowed else "resolved"
    rows = await db.execute_returning(
        "UPDATE censorship_reports SET status = :s, updated_at = NOW() "
        "WHERE id = :i RETURNING id, status",
        {"s": new_status, "i": report_id},
    )
    if not rows:
        raise ProblemError(404, "not_found", "report not found")
    return {"id": rows[0]["id"], "status": rows[0]["status"]}


async def _service_status(name: str) -> dict[str, Any] | None:
    from .. import db

    row = await db.fetch_one("SELECT * FROM service_status WHERE name = :n", {"n": name})
    if not row:
        return None
    return {
        "name": row["name"],
        "status": row["status"],
        "message": row.get("message"),
        "updatedAt": row["updated_at"].isoformat() if row.get("updated_at") else None,
    }


# ── App releases (in-app updater) ──────────────────────────────────────────
# Parity input #1: upload the APK through the admin API. Input #2 is a manual
# scp into data/app-releases/inbox/ (see services/releases.py) — both feed the
# same /v1/app-release/latest endpoint.


@router.get("/app-releases")
async def list_app_releases(admin: dict = Depends(require_admin)) -> dict[str, Any]:
    from ..services import releases

    return {"items": await releases.list_releases()}


@router.post("/app-releases", status_code=201)
async def upload_app_release(
    request: Request,
    versionCode: int = Query(...),
    versionName: str = Query(...),
    notes: str | None = Query(None),
    admin: dict = Depends(require_admin),
) -> dict[str, Any]:
    """Upload a client APK. Body = raw APK bytes (curl --data-binary @app.apk)."""
    from ..services import releases

    data = await request.body()
    if not data:
        raise ProblemError(400, "validation_failed", "empty APK body")
    return await releases.store_release(data, versionCode, versionName, notes)


# ── External embed cards (Yandex-style iframe integrations) ──────────────────


@router.get("/embeds/config")
async def get_embed_config(admin: dict = Depends(require_admin)) -> dict[str, Any]:
    from ..config import settings

    return {"edgeBaseUrl": settings.embed_edge_base_url}


@router.get("/embeds/search")
async def search_embed_entities(
    kind: Literal["track", "album", "artist"] = Query("track"),
    q: str = Query("", max_length=200),
    limit: int = Query(20, ge=1, le=50),
    admin: dict = Depends(require_admin),
) -> dict[str, Any]:
    from ..services import embed_service

    return {"items": await embed_service.search_entities_for_embed(kind, q, limit)}


@router.get("/embeds")
async def list_embeds(admin: dict = Depends(require_admin)) -> dict[str, Any]:
    from ..services import embed_service

    return {"items": await embed_service.list_widgets_admin()}


@router.post("/embeds", status_code=201)
async def create_embed(
    body: EmbedCreateRequest,
    admin: dict = Depends(require_admin),
) -> dict[str, Any]:
    from ..services import embed_service

    item = await embed_service.create_widget(
        kind=body.kind,
        entity_id=body.entityId,
        admin_user_id=admin["id"],
        title_override=body.title,
        theme=body.theme,
        layout=body.layout,
        allowed_origins=body.allowedOrigins,
        edge_base_url=body.edgeBaseUrl,
    )
    return {"item": item}


@router.get("/embeds/{widget_id}")
async def get_embed(
    widget_id: str,
    admin: dict = Depends(require_admin),
) -> dict[str, Any]:
    from ..services import embed_service

    return {"item": await embed_service.get_widget_admin(widget_id)}


@router.patch("/embeds/{widget_id}")
async def update_embed(
    widget_id: str,
    body: EmbedUpdateRequest,
    admin: dict = Depends(require_admin),
) -> dict[str, Any]:
    from ..services import embed_service

    item = await embed_service.update_widget(
        widget_id,
        title=body.title,
        theme=body.theme,
        layout=body.layout,
        allowed_origins=body.allowedOrigins,
        edge_base_url=body.edgeBaseUrl,
        active=body.active,
    )
    return {"item": item}


@router.post("/embeds/{widget_id}/warm")
async def warm_embed(
    widget_id: str,
    body: EmbedWarmRequest | None = None,
    admin: dict = Depends(require_admin),
) -> dict[str, Any]:
    from ..services import embed_service

    force = body.force if body is not None else True
    return {"item": await embed_service.trigger_widget_warm(widget_id, force=force)}


@router.delete("/embeds/{widget_id}", status_code=204)
async def delete_embed(
    widget_id: str,
    admin: dict = Depends(require_admin),
) -> Response:
    from ..services import embed_service

    if not await embed_service.delete_widget(widget_id):
        raise ProblemError(404, "not_found", "Интеграция не найдена")
    return Response(status_code=204)
