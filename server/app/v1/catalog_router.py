"""KWhale v1 tracks, albums, artists and media routes."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Header, Query, Request, Response

from .. import db
from ..auth import current_user, media_access
from ..errors import ProblemError
from ..services import albums as albums_service
from ..services import artists as artists_service
from ..services import dislikes as dislikes_service
from ..services import favorites as favorites_service
from ..services import providers, stream_service
from .deps import idempotent_get, idempotent_put, make_cursor, page_params, parse_id
from .likes import like_all
from .schemas import (
    Album,
    AlbumsPage,
    Artist,
    CensorshipReportRequest,
    CensorshipReportResponse,
    LikeAllResult,
    Lyrics,
    Track,
    TrackInfo,
    TrackQualityInfo,
    TrackSourceInfo,
    TracksPage,
)
from .serializers import (
    album_dto_by_ids,
    build_album_dto,
    build_artist_dto,
    build_lyrics,
    build_track_dtos,
    track_dto,
)

router = APIRouter(tags=["catalog"])


async def _track_ids_from_page(page: dict | None) -> list[int]:
    if not page:
        return []
    return [int(item["id"]) for item in page.get("items") or []]


@router.get("/tracks/{trackId}", response_model=Track, tags=["tracks"])
async def get_track(trackId: str, user: dict = Depends(current_user)) -> Track:
    item = await track_dto(parse_id(trackId, "trackId"), user["id"])
    if item is None:
        raise ProblemError(404, "not_found", "Track not found")
    return item


@router.get("/tracks/{trackId}/info", response_model=TrackInfo, tags=["tracks"])
async def get_track_info(trackId: str, user: dict = Depends(current_user)) -> TrackInfo:
    """Metadata + audio quality + sources for the player's «О треке» sheet."""
    tid = parse_id(trackId, "trackId")
    dtos = await build_track_dtos([tid], user["id"])
    if not dtos:
        raise ProblemError(404, "not_found", "Track not found")
    row = await db.fetch_one("SELECT isrc FROM tracks WHERE id = :i", {"i": tid})
    quality_row = await db.fetch_one(
        """
        SELECT codec, bitrate_kbps, sample_rate, bit_depth, channels, lossless,
               tier, bytes, duration_sec, error
        FROM track_quality WHERE track_id = :i
        """,
        {"i": tid},
    )
    source_rows = await db.fetch_all(
        """
        SELECT connector, available, quality_tier
        FROM track_sources WHERE track_id = :i
        ORDER BY CASE WHEN connector = 'local' THEN 0 ELSE 1 END,
                 quality_tier DESC NULLS LAST, connector
        """,
        {"i": tid},
    )
    quality = None
    if quality_row:
        quality = TrackQualityInfo(
            codec=quality_row.get("codec"),
            bitrate_kbps=quality_row.get("bitrate_kbps"),
            sample_rate=quality_row.get("sample_rate"),
            bit_depth=quality_row.get("bit_depth"),
            channels=quality_row.get("channels"),
            lossless=bool(quality_row.get("lossless")),
            tier=int(quality_row.get("tier") or 0),
            bytes=quality_row.get("bytes"),
            measured_duration_ms=(
                int(round(float(quality_row["duration_sec"]) * 1000))
                if quality_row.get("duration_sec") is not None
                else None
            ),
            error=quality_row.get("error"),
        )
    return TrackInfo(
        track=dtos[0],
        isrc=(row or {}).get("isrc"),
        quality=quality,
        sources=[
            TrackSourceInfo(
                connector=r["connector"],
                available=bool(r.get("available")),
                quality_tier=r.get("quality_tier"),
            )
            for r in source_rows
        ],
    )


@router.api_route("/tracks/{trackId}/stream", methods=["GET", "HEAD"], tags=["tracks"])
async def stream_track(
    trackId: str,
    request: Request,
    range_header: Annotated[str | None, Header(alias="Range")] = None,
    user: dict = Depends(media_access),
) -> Response:
    signed = bool(getattr(request.state, "media_signed", False))
    cache_control = "public, max-age=3600" if signed else "private, max-age=3600"
    return await stream_service.stream_response(
        parse_id(trackId, "trackId"),
        range_header,
        head_only=(request.method == "HEAD"),
        cache_control=cache_control,
    )


@router.get("/tracks/{trackId}/download", tags=["tracks"])
async def download_track(trackId: str, user: dict = Depends(current_user)) -> Response:
    return await stream_service.download_response(parse_id(trackId, "trackId"))


@router.get("/tracks/{trackId}/lyrics", response_model=Lyrics, tags=["tracks"])
async def get_lyrics(trackId: str, user: dict = Depends(current_user)) -> Lyrics:
    tid = parse_id(trackId, "trackId")
    row = await db.fetch_one(
        "SELECT title, artists, album, duration_sec FROM tracks WHERE id = :i", {"i": tid}
    )
    if not row:
        raise ProblemError(404, "not_found", "Track not found")
    artist = (row["artists"] or [""])[0]
    got = await providers.get_lyrics(
        artist, row["title"], row["duration_sec"] or 0, row.get("album"), track_id=tid
    )
    return build_lyrics(tid, got)


@router.put("/tracks/{trackId}/like", status_code=204, tags=["tracks"])
async def like_track(trackId: str, user: dict = Depends(current_user)) -> Response:
    result = await favorites_service.like(user["id"], parse_id(trackId, "trackId"))
    if not result.get("ok"):
        raise ProblemError(404, "not_found", "Track not found")
    return Response(status_code=204)


@router.delete("/tracks/{trackId}/like", status_code=204, tags=["tracks"])
async def unlike_track(trackId: str, user: dict = Depends(current_user)) -> Response:
    await favorites_service.unlike(user["id"], parse_id(trackId, "trackId"))
    return Response(status_code=204)


@router.put("/tracks/{trackId}/dislike", status_code=204, tags=["tracks"])
async def dislike_track(trackId: str, user: dict = Depends(current_user)) -> Response:
    ok = await dislikes_service.set_disliked(
        user["id"], parse_id(trackId, "trackId"), True
    )
    if not ok:
        raise ProblemError(404, "not_found", "Track not found")
    return Response(status_code=204)


@router.delete("/tracks/{trackId}/dislike", status_code=204, tags=["tracks"])
async def undislike_track(trackId: str, user: dict = Depends(current_user)) -> Response:
    ok = await dislikes_service.set_disliked(
        user["id"], parse_id(trackId, "trackId"), False
    )
    if not ok:
        raise ProblemError(404, "not_found", "Track not found")
    return Response(status_code=204)


@router.post(
    "/tracks/{trackId}/censorship",
    response_model=CensorshipReportResponse,
    tags=["tracks"],
)
async def report_censorship(
    trackId: str,
    body: CensorshipReportRequest,
    idempotency_key: Annotated[str | None, Header(alias="Idempotency-Key")] = None,
    user: dict = Depends(current_user),
) -> CensorshipReportResponse:
    """Flag a track for admin censorship review (client: «Отправить на цензуру»).

    Idempotent per (user, track): re-sending returns the existing open report.
    """
    tid = parse_id(trackId, "trackId")
    cached = await idempotent_get(user["id"], idempotency_key, f"censorship:{tid}")
    if cached:
        return CensorshipReportResponse(**cached)
    track = await db.fetch_one("SELECT id FROM tracks WHERE id = :i", {"i": tid})
    if not track:
        raise ProblemError(404, "not_found", "Track not found")
    rows = await db.execute_returning(
        """
        INSERT INTO censorship_reports (user_id, track_id, reason, comment, position_ms)
        VALUES (:u, :t, :r, :c, :p)
        ON CONFLICT (user_id, track_id) DO UPDATE
        SET reason = EXCLUDED.reason,
            comment = EXCLUDED.comment,
            position_ms = EXCLUDED.position_ms,
            status = 'pending',
            updated_at = NOW()
        RETURNING id, status
        """,
        {
            "u": user["id"],
            "t": tid,
            "r": (body.reason or "censored").strip()[:64],
            "c": (body.comment or None) and body.comment.strip()[:1000],
            "p": body.position_ms,
        },
    )
    row = rows[0]
    result = CensorshipReportResponse(
        ok=True, report_id=str(row["id"]), status=row["status"]
    )
    await idempotent_put(
        user["id"],
        idempotency_key,
        f"censorship:{tid}",
        result.model_dump(by_alias=True),
    )
    return result


@router.delete("/tracks/{trackId}/censorship", status_code=204, tags=["tracks"])
async def cancel_censorship_report(
    trackId: str, user: dict = Depends(current_user)
) -> Response:
    """Cancel the user's open censorship report for this track."""
    await db.execute(
        "DELETE FROM censorship_reports WHERE user_id = :u AND track_id = :t",
        {"u": user["id"], "t": parse_id(trackId, "trackId")},
    )
    return Response(status_code=204)


@router.get("/albums/{albumId}", response_model=Album, tags=["albums"])
async def get_album(albumId: str, user: dict = Depends(current_user)) -> Album:
    detail = await albums_service.get_album(parse_id(albumId, "albumId"), user["id"])
    if not detail:
        raise ProblemError(404, "not_found", "Album not found")
    ids = await _track_ids_from_page(detail.get("tracks"))
    return await build_album_dto(detail["album"], tracks=await build_track_dtos(ids, user["id"]))


@router.put("/albums/{albumId}/like", status_code=204, tags=["albums"])
async def like_album(albumId: str, user: dict = Depends(current_user)) -> Response:
    """Save the album card to the library. No track likes, no downloads."""
    result = await favorites_service.like_album(user["id"], parse_id(albumId, "albumId"))
    if not result.get("ok"):
        raise ProblemError(404, "not_found", "Album not found")
    return Response(status_code=204)


@router.delete("/albums/{albumId}/like", status_code=204, tags=["albums"])
async def unlike_album(albumId: str, user: dict = Depends(current_user)) -> Response:
    await favorites_service.unlike_album(user["id"], parse_id(albumId, "albumId"))
    return Response(status_code=204)


@router.get("/albums/{albumId}/tracks", response_model=TracksPage, tags=["albums"])
async def get_album_tracks(
    albumId: str,
    cursor: str | None = Query(None),
    limit: int | None = Query(None, ge=1, le=100),
    user: dict = Depends(current_user),
) -> TracksPage:
    detail = await albums_service.get_album(parse_id(albumId, "albumId"), user["id"])
    if not detail:
        raise ProblemError(404, "not_found", "Album not found")
    ids = await _track_ids_from_page(detail.get("tracks"))
    lim, offset = page_params(cursor, limit)
    chunk = ids[offset : offset + lim]
    return TracksPage(
        items=await build_track_dtos(chunk, user["id"]),
        next_cursor=make_cursor(offset, len(ids), lim),
    )


@router.put(
    "/albums/{albumId}/tracks/like",
    response_model=LikeAllResult,
    status_code=202,
    tags=["albums"],
)
async def like_album_tracks(
    albumId: str,
    idempotency_key: Annotated[str | None, Header(alias="Idempotency-Key")] = None,
    user: dict = Depends(current_user),
) -> LikeAllResult:
    aid = parse_id(albumId, "albumId")
    cached = await idempotent_get(user["id"], idempotency_key, f"album-like:{aid}")
    if cached:
        return LikeAllResult(**cached)
    detail = await albums_service.get_album(aid, user["id"])
    if not detail:
        raise ProblemError(404, "not_found", "Album not found")
    ids = await _track_ids_from_page(detail.get("tracks"))
    result = await like_all(ids, user["id"])
    await idempotent_put(
        user["id"], idempotency_key, f"album-like:{aid}", result.model_dump(by_alias=True)
    )
    return result


@router.get("/artists/{artistId}", response_model=Artist, tags=["artists"])
async def get_artist(artistId: str, user: dict = Depends(current_user)) -> Artist:
    detail = await artists_service.get_artist(parse_id(artistId, "artistId"), user_id=user["id"])
    if not detail:
        raise ProblemError(404, "not_found", "Artist not found")
    top_ids = await _track_ids_from_page(detail.get("top_tracks"))
    album_ids = [int(a["id"]) for a in (detail.get("albums") or {}).get("items") or []]
    latest = detail.get("latest_release")
    return await build_artist_dto(
        detail["artist"],
        albums=await album_dto_by_ids(album_ids),
        top_tracks=await build_track_dtos(top_ids, user["id"]),
        latest_release=await build_album_dto(latest) if latest else None,
    )


@router.put("/artists/{artistId}/like", status_code=204, tags=["artists"])
async def like_artist(artistId: str, user: dict = Depends(current_user)) -> Response:
    """Save the artist card to the library. No downloads."""
    aid = parse_id(artistId, "artistId")
    result = await favorites_service.like_artist(user["id"], aid)
    if not result.get("ok"):
        raise ProblemError(404, "not_found", "Artist not found")
    # Persist the discography in the background; the like response stays fast.
    artists_service.schedule_artist_sync(aid)
    return Response(status_code=204)


@router.delete("/artists/{artistId}/like", status_code=204, tags=["artists"])
async def unlike_artist(artistId: str, user: dict = Depends(current_user)) -> Response:
    await favorites_service.unlike_artist(user["id"], parse_id(artistId, "artistId"))
    return Response(status_code=204)


@router.post("/artists/{artistId}/sync", status_code=202, tags=["artists"])
async def sync_artist_now(artistId: str, user: dict = Depends(current_user)) -> dict:
    """Kick off a full discography sync; poll GET .../sync for status."""
    aid = parse_id(artistId, "artistId")
    if not await db.fetch_one("SELECT 1 FROM artists WHERE id = :i", {"i": aid}):
        raise ProblemError(404, "not_found", "Artist not found")
    artists_service.schedule_artist_sync(aid)
    return await artists_service.artist_sync_status(aid)


@router.get("/artists/{artistId}/sync", tags=["artists"])
async def get_artist_sync(artistId: str, user: dict = Depends(current_user)) -> dict:
    aid = parse_id(artistId, "artistId")
    if not await db.fetch_one("SELECT 1 FROM artists WHERE id = :i", {"i": aid}):
        raise ProblemError(404, "not_found", "Artist not found")
    return await artists_service.artist_sync_status(aid)


@router.get("/artists/{artistId}/tracks", response_model=TracksPage, tags=["artists"])
async def get_artist_tracks(
    artistId: str,
    cursor: str | None = Query(None),
    limit: int | None = Query(None, ge=1, le=100),
    user: dict = Depends(current_user),
) -> TracksPage:
    """All known tracks of the artist (DB), lifetime-independent of top-20."""
    lim, offset = page_params(cursor, limit)
    detail = await artists_service.list_artist_tracks(
        parse_id(artistId, "artistId"), lim, offset, user["id"]
    )
    if detail is None:
        raise ProblemError(404, "not_found", "Artist not found")
    ids = [int(item["id"]) for item in detail["items"]]
    return TracksPage(
        items=await build_track_dtos(ids, user["id"]),
        next_cursor=make_cursor(offset, int(detail.get("total") or 0), lim),
        total=detail["total"],
    )


@router.get("/artists/{artistId}/albums", response_model=AlbumsPage, tags=["artists"])
async def get_artist_albums(
    artistId: str,
    cursor: str | None = Query(None),
    limit: int | None = Query(None, ge=1, le=100),
    user: dict = Depends(current_user),
) -> AlbumsPage:
    detail = await artists_service.get_artist_albums(parse_id(artistId, "artistId"))
    if not detail:
        raise ProblemError(404, "not_found", "Artist not found")
    ids = [int(a["id"]) for a in (detail.get("albums") or {}).get("items") or []]
    lim, offset = page_params(cursor, limit)
    chunk = ids[offset : offset + lim]
    return AlbumsPage(
        items=await album_dto_by_ids(chunk),
        next_cursor=make_cursor(offset, len(ids), lim),
    )


@router.get("/artists/{artistId}/top-tracks", response_model=TracksPage, tags=["artists"])
async def get_artist_top_tracks(
    artistId: str,
    limit: int | None = Query(None, ge=1, le=100),
    user: dict = Depends(current_user),
) -> TracksPage:
    detail = await artists_service.get_artist(parse_id(artistId, "artistId"), user_id=user["id"])
    if not detail:
        raise ProblemError(404, "not_found", "Artist not found")
    ids = await _track_ids_from_page(detail.get("top_tracks"))
    lim = limit or 50
    return TracksPage(items=await build_track_dtos(ids[:lim], user["id"]), next_cursor=None)
