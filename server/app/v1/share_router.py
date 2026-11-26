"""Authenticated KWhale Core v1 share creation & lookup routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from ..auth import current_user
from ..config import settings
from ..errors import ProblemError
from ..services import share_service
from .deps import parse_id
from .schemas import ShareCreateRequest, ShareLink, SharePlatformLink

router = APIRouter(prefix="/share", tags=["share"])


def _to_dto(rec: share_service.ShareRecord, origin: str) -> ShareLink:
    return ShareLink(
        id=rec.id,
        kind=rec.kind,
        entity_id=str(rec.entity_id),
        title=rec.title,
        subtitle=rec.subtitle,
        meta=rec.meta,
        image_id=rec.image_id,
        share_url=f"{origin}/share/s/{rec.id}",
        card_url=f"{origin}/share/card/{rec.id}.png",
        cover_url=f"{origin}/share/cover/{rec.id}",
        deep_link=share_service.deep_link_for(rec.kind, rec.entity_id),
        expires_at=rec.expires_at.isoformat(),
        ttl_days=settings.share_link_ttl_days,
        clicks_total=rec.clicks_total,
        platforms=[
            SharePlatformLink(
                platform=p.platform,
                label=p.label,
                url=p.url,
                go_url=f"{origin}/share/go/{rec.id}/{p.platform}",
                exact=p.exact,
            )
            for p in rec.platforms
        ],
    )


@router.post("", response_model=ShareLink)
async def create_share(
    body: ShareCreateRequest,
    request: Request,
    user: dict = Depends(current_user),
) -> ShareLink:
    """Create or renew the canonical short share link (7-day TTL) for a track, album or artist."""
    entity_id = parse_id(body.id, f"{body.kind}Id")
    rec = await share_service.create_or_refresh_share(body.kind, entity_id, user["id"])
    return _to_dto(rec, share_service.request_origin(request))


@router.get("/{shareId}", response_model=ShareLink)
async def get_share_details(
    shareId: str,
    request: Request,
    user: dict = Depends(current_user),
) -> ShareLink:
    """Resolve an active share link id back to KWhale entity metadata."""
    try:
        rec = await share_service.get_share(shareId)
    except share_service.ShareExpiredError:
        raise ProblemError(410, "share_expired", "Share link has expired") from None
    except share_service.ShareNotFoundError:
        raise ProblemError(404, "not_found", "Share link not found") from None
    return _to_dto(rec, share_service.request_origin(request))
