"""KWhale v1 playlists routes."""

from __future__ import annotations

from typing import Annotated, Literal

from fastapi import APIRouter, Depends, File, Header, Query, Response, UploadFile

from .. import db
from ..auth import current_user
from ..errors import ProblemError
from ..services import playlists as playlists_service
from . import images
from .deps import idempotent_get, idempotent_put, make_cursor, page_params, parse_id
from .likes import like_all
from .schemas import (
    LikeAllResult,
    Playlist,
    PlaylistCreate,
    PlaylistReorder,
    PlaylistsPage,
    PlaylistTracksAdd,
    PlaylistUpdate,
    TracksPage,
)
from .serializers import build_playlist_dto, build_track_dtos

router = APIRouter(prefix="/playlists", tags=["playlists"])


async def _playlist_with_tracks(row: dict, user_id: int, limit: int = 200) -> Playlist:
    page = await playlists_service.list_tracks(row["id"], user_id, limit, 0)
    ids = [int(t["id"]) for t in (page or {}).get("items") or []]
    return await build_playlist_dto(
        row, user_id, tracks=await build_track_dtos(ids, user_id)
    )


@router.get("", response_model=PlaylistsPage)
async def list_playlists(
    owner: Literal["me", "all"] = "me",
    cursor: str | None = Query(None),
    limit: int | None = Query(None, ge=1, le=100),
    user: dict = Depends(current_user),
) -> PlaylistsPage:
    lim, offset = page_params(cursor, limit)
    if owner == "all":
        page = await playlists_service.list_visible(user["id"], lim, offset)
    else:
        page = await playlists_service.list_playlists(user["id"], lim, offset, "own")
    items = [await build_playlist_dto(r, user["id"]) for r in page.get("items") or []]
    return PlaylistsPage(
        items=items,
        next_cursor=make_cursor(offset, int(page.get("total") or 0), lim),
    )


@router.post("", response_model=Playlist, status_code=201)
async def create_playlist(
    body: PlaylistCreate,
    idempotency_key: Annotated[str | None, Header(alias="Idempotency-Key")] = None,
    user: dict = Depends(current_user),
) -> Playlist:
    cached = await idempotent_get(user["id"], idempotency_key, "playlist-create")
    if cached:
        return Playlist(**cached)
    row = await playlists_service.create(
        user["id"], body.title, body.description, body.visibility
    )
    if not row:
        raise ProblemError(500, "internal_error", "Playlist creation failed")
    dto = await build_playlist_dto(row, user["id"])
    await idempotent_put(
        user["id"], idempotency_key, "playlist-create", dto.model_dump(by_alias=True)
    )
    return dto


@router.get("/{playlistId}", response_model=Playlist)
async def get_playlist(playlistId: str, user: dict = Depends(current_user)) -> Playlist:
    pid = parse_id(playlistId, "playlistId")
    row = await playlists_service.get_meta(pid, user["id"])
    if not row:
        raise ProblemError(404, "not_found", "Playlist not found")
    return await _playlist_with_tracks(row, user["id"])


@router.patch("/{playlistId}", response_model=Playlist)
async def update_playlist(
    playlistId: str, body: PlaylistUpdate, user: dict = Depends(current_user)
) -> Playlist:
    pid = parse_id(playlistId, "playlistId")
    if not await playlists_service.exists(pid):
        raise ProblemError(404, "not_found", "Playlist not found")
    row = await playlists_service.update(
        pid, user["id"], body.title, body.description, body.visibility
    )
    if not row:
        raise ProblemError(403, "forbidden", "Playlist not owned")
    return await _playlist_with_tracks(row, user["id"])


@router.delete("/{playlistId}", status_code=204)
async def delete_playlist(playlistId: str, user: dict = Depends(current_user)) -> Response:
    pid = parse_id(playlistId, "playlistId")
    if not await playlists_service.exists(pid):
        raise ProblemError(404, "not_found", "Playlist not found")
    ok = await playlists_service.delete(pid, user["id"])
    if not ok:
        raise ProblemError(403, "forbidden", "Playlist not owned")
    return Response(status_code=204)


@router.get("/{playlistId}/tracks", response_model=TracksPage)
async def get_playlist_tracks(
    playlistId: str,
    cursor: str | None = Query(None),
    limit: int | None = Query(None, ge=1, le=100),
    user: dict = Depends(current_user),
) -> TracksPage:
    lim, offset = page_params(cursor, limit)
    page = await playlists_service.list_tracks(
        parse_id(playlistId, "playlistId"), user["id"], lim, offset
    )
    if page is None:
        raise ProblemError(404, "not_found", "Playlist not found")
    ids = [int(t["id"]) for t in page.get("items") or []]
    return TracksPage(
        items=await build_track_dtos(ids, user["id"]),
        next_cursor=make_cursor(offset, int(page.get("total") or 0), lim),
    )


@router.post("/{playlistId}/tracks", response_model=Playlist)
async def add_playlist_tracks(
    playlistId: str,
    body: PlaylistTracksAdd,
    idempotency_key: Annotated[str | None, Header(alias="Idempotency-Key")] = None,
    user: dict = Depends(current_user),
) -> Playlist:
    pid = parse_id(playlistId, "playlistId")
    cached = await idempotent_get(user["id"], idempotency_key, f"playlist-add:{pid}")
    if cached:
        return Playlist(**cached)
    track_ids = [parse_id(t, "trackId") for t in body.track_ids]
    if not await playlists_service.exists(pid):
        raise ProblemError(404, "not_found", "Playlist not found")
    row = await playlists_service.add_tracks(pid, user["id"], track_ids, body.position)
    if row is None:
        raise ProblemError(403, "forbidden", "Playlist not owned")
    dto = await _playlist_with_tracks(row, user["id"])
    await idempotent_put(
        user["id"], idempotency_key, f"playlist-add:{pid}", dto.model_dump(by_alias=True)
    )
    return dto


@router.delete("/{playlistId}/tracks/{trackId}", status_code=204)
async def remove_playlist_track(
    playlistId: str, trackId: str, user: dict = Depends(current_user)
) -> Response:
    pid = parse_id(playlistId, "playlistId")
    tid = parse_id(trackId, "trackId")
    if not await playlists_service.exists(pid):
        raise ProblemError(404, "not_found", "Playlist not found")
    if not await playlists_service._owns(pid, user["id"]):
        raise ProblemError(403, "forbidden", "Playlist not owned")
    ok = await playlists_service.remove_track(pid, user["id"], tid)
    if not ok:
        raise ProblemError(404, "not_found", "Track not found in playlist")
    return Response(status_code=204)


@router.put("/{playlistId}/tracks/order", response_model=Playlist)
async def reorder_playlist_tracks(
    playlistId: str, body: PlaylistReorder, user: dict = Depends(current_user)
) -> Playlist:
    pid = parse_id(playlistId, "playlistId")
    if not await playlists_service.exists(pid):
        raise ProblemError(404, "not_found", "Playlist not found")
    ids = [parse_id(t, "trackId") for t in body.track_ids]
    ok = await playlists_service.reorder(pid, user["id"], ids)
    if not ok:
        raise ProblemError(403, "forbidden", "Playlist not owned")
    row = await playlists_service.get_meta(pid, user["id"])
    if not row:
        raise ProblemError(404, "not_found", "Playlist not found")
    return await _playlist_with_tracks(row, user["id"])


@router.put("/{playlistId}/follow", status_code=204)
async def follow_playlist(playlistId: str, user: dict = Depends(current_user)) -> Response:
    result = await playlists_service.follow(parse_id(playlistId, "playlistId"), user["id"])
    if not result.get("ok"):
        if result.get("error") == "own_playlist":
            raise ProblemError(400, "validation_failed", "Cannot follow your own playlist")
        raise ProblemError(404, "not_found", "Playlist not found")
    return Response(status_code=204)


@router.delete("/{playlistId}/follow", status_code=204)
async def unfollow_playlist(playlistId: str, user: dict = Depends(current_user)) -> Response:
    await playlists_service.unfollow(parse_id(playlistId, "playlistId"), user["id"])
    return Response(status_code=204)


@router.post("/{playlistId}/cover", response_model=Playlist)
async def upload_playlist_cover(
    playlistId: str,
    file: UploadFile = File(...),
    user: dict = Depends(current_user),
) -> Playlist:
    """Upload a user cover; only the playlist owner can change it."""
    pid = parse_id(playlistId, "playlistId")
    row = await playlists_service.get_meta(pid, user["id"])
    if not row:
        raise ProblemError(404, "not_found", "Playlist not found")
    if int(row.get("user_id") or 0) != int(user["id"]):
        raise ProblemError(403, "forbidden", "Playlist not owned")
    mime = (file.content_type or "").split(";")[0]
    if not mime.startswith("image/"):
        raise ProblemError(415, "unsupported_media_type", "Cover must be an image")
    data = await file.read()
    if not data:
        raise ProblemError(400, "validation_failed", "Empty image")
    if len(data) > 10 * 1024 * 1024:
        raise ProblemError(413, "payload_too_large", "Cover is larger than 10 MB")
    image_id = await images.store_uploaded_image(data, mime)
    await db.execute(
        "UPDATE playlists SET cover_image_id = :c, updated_at = NOW() WHERE id = :p",
        {"c": image_id, "p": pid},
    )
    row = await playlists_service.get_meta(pid, user["id"])
    return await _playlist_with_tracks(row, user["id"])


@router.delete("/{playlistId}/cover", status_code=204)
async def delete_playlist_cover(playlistId: str, user: dict = Depends(current_user)) -> Response:
    pid = parse_id(playlistId, "playlistId")
    row = await playlists_service.get_meta(pid, user["id"])
    if not row:
        raise ProblemError(404, "not_found", "Playlist not found")
    if int(row.get("user_id") or 0) != int(user["id"]):
        raise ProblemError(403, "forbidden", "Playlist not owned")
    await db.execute(
        "UPDATE playlists SET cover_image_id = NULL, updated_at = NOW() WHERE id = :p",
        {"p": pid},
    )
    return Response(status_code=204)


@router.put(
    "/{playlistId}/tracks/like",
    response_model=LikeAllResult,
    status_code=202,
)
async def like_playlist_tracks(
    playlistId: str,
    idempotency_key: Annotated[str | None, Header(alias="Idempotency-Key")] = None,
    user: dict = Depends(current_user),
) -> LikeAllResult:
    pid = parse_id(playlistId, "playlistId")
    cached = await idempotent_get(user["id"], idempotency_key, f"playlist-like:{pid}")
    if cached:
        return LikeAllResult(**cached)
    page = await playlists_service.list_tracks(pid, user["id"], 10000, 0)
    if page is None:
        raise ProblemError(404, "not_found", "Playlist not found")
    ids = [int(t["id"]) for t in page.get("items") or []]
    result = await like_all(ids, user["id"])
    await idempotent_put(
        user["id"], idempotency_key, f"playlist-like:{pid}", result.model_dump(by_alias=True)
    )
    return result
