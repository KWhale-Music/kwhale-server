"""KWhale v1 image endpoint."""

from __future__ import annotations

from fastapi import APIRouter, Header, Query, Response

from ..errors import ProblemError
from .images import get_image

router = APIRouter(prefix="/images", tags=["images"])


@router.get("/{imageId}")
async def image(
    imageId: str,
    size: str = Query("md"),
    if_none_match: str | None = Header(None, alias="If-None-Match"),
) -> Response:
    if size not in ("xs", "sm", "md", "lg", "original"):
        raise ProblemError(400, "validation_failed", "invalid image size")
    response = await get_image(imageId, size)
    etag = response.headers.get("ETag")
    if etag and if_none_match == etag:
        return Response(status_code=304, headers={"ETag": etag})
    return response
