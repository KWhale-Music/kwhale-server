"""KWhale v1 home, library, history, recommendations, mixes, events, config."""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime

from fastapi import APIRouter, Depends, Query, Response

from .. import db
from ..auth import current_user, require_admin
from ..config import settings
from ..errors import ProblemError
from ..services import albums as albums_service
from ..services import collections as collections_service
from ..services import playlists as playlists_service
from ..services.rec import llm as rec_llm
from ..services.rec import mixes as rec_mixes
from .deps import make_cursor, page_params
from .schemas import (
    AlbumsPage,
    Artist,
    ArtistRef,
    ArtistsPage,
    Config,
    EventBatch,
    GenresResponse,
    HistoryEntry,
    HistoryPage,
    HomeItem,
    HomeResponse,
    HomeSection,
    Mix,
    MixesPage,
    Onboarding,
    PlaylistsPage,
    PromptMixRequest,
    TrackRecommendation,
    TrackRecommendationsPage,
    TracksPage,
)
from .serializers import (
    album_dto_by_ids,
    build_artist_dto,
    build_playlist_dto,
    build_track_dtos,
    genre_from_name,
)

router = APIRouter()


@router.get("/home", response_model=HomeResponse)
async def home(user: dict = Depends(current_user)) -> HomeResponse:
    recent_rows = await db.fetch_all(
        """
        SELECT track_id
        FROM events
        WHERE user_id = :u AND type = 'play' AND track_id IS NOT NULL
        ORDER BY occurred_at DESC NULLS LAST, received_at DESC
        LIMIT 60
        """,
        {"u": user["id"]},
    )
    recent_ids: list[int] = []
    seen_recent: set[int] = set()
    for r in recent_rows:
        tid = int(r["track_id"])
        if tid not in seen_recent:
            seen_recent.add(tid)
            recent_ids.append(tid)
            if len(recent_ids) >= 20:
                break
    recent_tracks = await build_track_dtos(recent_ids, user["id"]) if recent_ids else []

    # Home is discovery-only. Library-owned blocks (liked tracks, playlists,
    # saved albums) are served by /v1/library/*; recommendation cards come from
    # /v1/mixes. Sending them here duplicated the same rows on two screens.
    sections: list[HomeSection] = []
    if recent_tracks:
        sections.append(
            HomeSection(
                id="recent",
                kind="shelf",
                layout="carousel",
                title="Недавно прослушанное",
                items=[HomeItem(type="track", track=t) for t in recent_tracks],
            )
        )
    return HomeResponse(sections=sections)


@router.get("/library/tracks", response_model=TracksPage)
async def library_tracks(
    cursor: str | None = Query(None),
    limit: int | None = Query(None, ge=1, le=100),
    sort: str = Query("default"),
    q: str | None = Query(None, max_length=200),
    user: dict = Depends(current_user),
) -> TracksPage:
    lim, offset = page_params(cursor, limit)
    service_sort = {"default": "added", "added": "added", "title": "title", "artist": "artist"}.get(
        sort, "added"
    )
    page = await collections_service.favorites_page(user["id"], lim, offset, service_sort, q)
    ids = [int(t["id"]) for t in page.get("items") or []]
    return TracksPage(
        items=await build_track_dtos(ids, user["id"]),
        next_cursor=make_cursor(offset, int(page.get("total") or 0), lim),
        total=int(page.get("total") or 0),
    )


@router.get("/library/playlists", response_model=PlaylistsPage)
async def library_playlists(
    cursor: str | None = Query(None),
    limit: int | None = Query(None, ge=1, le=100),
    user: dict = Depends(current_user),
) -> PlaylistsPage:
    lim, offset = page_params(cursor, limit)
    page = await playlists_service.list_playlists(user["id"], lim, offset, "library")
    items = [await build_playlist_dto(r, user["id"]) for r in page.get("items") or []]
    return PlaylistsPage(
        items=items,
        next_cursor=make_cursor(offset, int(page.get("total") or 0), lim),
        total=int(page.get("total") or 0),
    )


@router.get("/library/albums", response_model=AlbumsPage)
async def library_albums(
    cursor: str | None = Query(None),
    limit: int | None = Query(None, ge=1, le=100),
    user: dict = Depends(current_user),
) -> AlbumsPage:
    """Only albums explicitly saved with the album card action."""
    lim, offset = page_params(cursor, limit)
    page = await albums_service.list_favorite_albums(user["id"], lim, offset)
    ids = [int(a["id"]) for a in page.get("items") or []]
    return AlbumsPage(
        items=await album_dto_by_ids(ids, user["id"]),
        next_cursor=make_cursor(offset, int(page.get("total") or 0), lim),
        total=int(page.get("total") or 0),
    )


@router.get("/history", response_model=HistoryPage)
async def history(
    cursor: str | None = Query(None),
    limit: int | None = Query(None, ge=1, le=100),
    user: dict = Depends(current_user),
) -> HistoryPage:
    lim, offset = page_params(cursor, limit)
    total_row = await db.fetch_one(
        "SELECT COUNT(*) AS n FROM events WHERE user_id = :u AND type = 'play' AND track_id IS NOT NULL",
        {"u": user["id"]},
    )
    rows = await db.fetch_all(
        """
        SELECT track_id, occurred_at, payload
        FROM events
        WHERE user_id = :u AND type = 'play' AND track_id IS NOT NULL
        ORDER BY occurred_at DESC NULLS LAST, received_at DESC
        LIMIT :l OFFSET :o
        """,
        {"u": user["id"], "l": lim, "o": offset},
    )
    track_ids = [int(r["track_id"]) for r in rows]
    tracks = await build_track_dtos(track_ids, user["id"])
    by_id = {int(t.id): t for t in tracks}
    items = []
    for r in rows:
        track = by_id.get(int(r["track_id"]))
        if track is None:
            continue
        payload = r.get("payload") or {}
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except ValueError:
                payload = {}
        items.append(
            HistoryEntry(
                track=track,
                played_at=r["occurred_at"].isoformat() if r.get("occurred_at") else "",
                context=payload.get("surface"),
            )
        )
    return HistoryPage(
        items=items,
        next_cursor=make_cursor(offset, int((total_row or {}).get("n") or 0), lim),
        total=int((total_row or {}).get("n") or 0),
    )


@router.delete("/history", status_code=204)
async def clear_history(user: dict = Depends(current_user)) -> Response:
    await db.execute(
        "DELETE FROM events WHERE user_id = :u AND type IN ('play','pause','resume','skip','complete')",
        {"u": user["id"]},
    )
    return Response(status_code=204)


@router.get("/recommendations/top-artists", response_model=ArtistsPage)
async def top_artists(
    limit: int | None = Query(None, ge=1, le=100),
    user: dict = Depends(current_user),
) -> ArtistsPage:
    rows = await db.fetch_all(
        """
        SELECT a.id, a.name, a.cover_url, a.genre, COUNT(*) AS plays
        FROM favorites f
        JOIN track_artists ta ON ta.track_id = f.track_id
        JOIN artists a ON a.id = ta.artist_id
        WHERE f.user_id = :u
        GROUP BY a.id, a.name, a.cover_url, a.genre
        ORDER BY plays DESC, lower(a.name)
        LIMIT :l
        """,
        {"u": user["id"], "l": limit or 20},
    )
    items: list[Artist] = []
    for r in rows:
        items.append(
            await build_artist_dto(
                {"id": r["id"], "name": r["name"], "cover_url": r["cover_url"], "genre": r["genre"]}
            )
        )
    return ArtistsPage(items=items, next_cursor=None)


@router.get("/recommendations/genres", response_model=GenresResponse)
async def genres(user: dict = Depends(current_user)) -> GenresResponse:
    rows = await db.fetch_all(
        """
        SELECT genre, COUNT(*) AS n FROM tracks
        WHERE genre IS NOT NULL AND genre <> ''
        GROUP BY genre ORDER BY n DESC, lower(genre) LIMIT 50
        """
    )
    items = [g for r in rows if (g := genre_from_name(r["genre"])) is not None]
    return GenresResponse(items=items)


@router.get("/recommendations/onboarding", response_model=Onboarding)
async def get_onboarding(user: dict = Depends(current_user)) -> Onboarding:
    row = await db.fetch_one(
        "SELECT genres, artist_ids FROM user_onboarding WHERE user_id = :u",
        {"u": user["id"]},
    )
    if not row:
        return Onboarding(genres=[], artists=[])
    genre_items = [
        g for raw in (row.get("genres") or []) if (g := genre_from_name(raw)) is not None
    ]
    raw_artist_ids = list(row.get("artist_ids") or [])
    int_ids = [int(aid) for aid in raw_artist_ids if str(aid).isdigit()]
    names_by_id: dict[str, str] = {}
    if int_ids:
        placeholders = [f":a{i}" for i in range(len(int_ids))]
        ar_rows = await db.fetch_all(
            f"SELECT id, name FROM artists WHERE id IN ({', '.join(placeholders)})",
            {f"a{i}": v for i, v in enumerate(int_ids)},
        )
        names_by_id = {str(r["id"]): r["name"] for r in ar_rows}
    artist_items = [
        ArtistRef(id=str(aid), name=names_by_id.get(str(aid), str(aid)))
        for aid in raw_artist_ids
    ]
    return Onboarding(genres=genre_items, artists=artist_items)


@router.put("/recommendations/onboarding", status_code=204)
async def update_onboarding(body: Onboarding, user: dict = Depends(current_user)) -> Response:
    genre_names = [g.name or g.id for g in body.genres if (g.name or g.id)]
    artist_ids = [a.id for a in body.artists if a.id]
    await db.execute(
        """
        INSERT INTO user_onboarding (user_id, genres, artist_ids, updated_at)
        VALUES (:u, :genres, :artists, NOW())
        ON CONFLICT (user_id) DO UPDATE
        SET genres = EXCLUDED.genres,
            artist_ids = EXCLUDED.artist_ids,
            updated_at = NOW()
        """,
        {"u": user["id"], "genres": genre_names, "artists": artist_ids},
    )
    return Response(status_code=204)


async def _recommendation_items(scored: list, user_id: int) -> list[TrackRecommendation]:
    if not scored:
        return []
    tracks = await build_track_dtos([item.candidate.track_id for item in scored], user_id)
    by_id = {int(track.id): track for track in tracks}
    items: list[TrackRecommendation] = []
    for item in scored:
        track = by_id.get(item.candidate.track_id)
        if track is None:
            continue
        items.append(
            TrackRecommendation(
                track=track,
                reason=rec_mixes.reason_text(item),
                reason_code=item.reason_code,
                score=item.score,
            )
        )
    return items


@router.get("/mixes", response_model=MixesPage)
async def mixes(
    kind: str = Query("personal"),
    cursor: str | None = Query(None),
    limit: int | None = Query(None, ge=1, le=100),
    user: dict = Depends(current_user),
) -> MixesPage:
    if kind != "personal":
        raise ProblemError(400, "validation_failed", "only personal mixes are supported")
    items = await rec_mixes.list_mixes(user["id"])
    return MixesPage(items=[Mix(**item) for item in items], next_cursor=None)


@router.get("/mixes/{mixId}/tracks", response_model=TrackRecommendationsPage)
async def mix_tracks(
    mixId: str,
    cursor: str | None = Query(None),
    limit: int | None = Query(None, ge=1, le=100),
    user: dict = Depends(current_user),
) -> TrackRecommendationsPage:
    lim, offset = page_params(cursor, limit)
    try:
        scored = await rec_mixes.build(user["id"], mixId, lim, offset)
    except rec_mixes.UnknownMix:
        raise ProblemError(404, "not_found", "Mix not found") from None
    return TrackRecommendationsPage(
        items=await _recommendation_items(scored, user["id"]),
        next_cursor=None,
    )


@router.post("/mixes/prompt", response_model=TrackRecommendationsPage)
async def prompt_mix(
    body: PromptMixRequest,
    user: dict = Depends(current_user),
) -> TrackRecommendationsPage:
    scored = await rec_mixes.prompt_mix(
        user["id"],
        body.prompt,
        body.size,
        api_key=body.llm_api_key,
        endpoint=body.llm_endpoint,
        model=body.llm_model,
    )
    return TrackRecommendationsPage(
        items=await _recommendation_items(scored, user["id"]),
        next_cursor=None,
    )


@router.get("/tracks/{trackId}/similar", response_model=TrackRecommendationsPage)
async def similar_tracks(
    trackId: str,
    limit: int | None = Query(None, ge=1, le=100),
    user: dict = Depends(current_user),
) -> TrackRecommendationsPage:
    try:
        track_id = int(trackId)
    except ValueError:
        raise ProblemError(404, "not_found", "Track not found") from None
    scored = await rec_mixes.similar(user["id"], track_id, limit or 40)
    return TrackRecommendationsPage(
        items=await _recommendation_items(scored, user["id"]),
        next_cursor=None,
    )


@router.post("/events", status_code=202)
async def send_events(body: EventBatch, user: dict = Depends(current_user)) -> Response:
    if not body.events:
        raise ProblemError(400, "validation_failed", "events must not be empty")
    rows = []
    for ev in body.events:
        try:
            uuid.UUID(ev.id)
        except ValueError:
            raise ProblemError(400, "validation_failed", f"invalid event id: {ev.id}") from None
        track_id = None
        if ev.track_id:
            try:
                track_id = int(ev.track_id)
            except ValueError:
                track_id = None
        payload = ev.model_dump(by_alias=True, exclude_none=True)
        try:
            occurred = datetime.fromisoformat(ev.occurred_at.replace("Z", "+00:00"))
        except ValueError:
            occurred = datetime.now(UTC)
        rows.append(
            {
                "id": ev.id,
                "u": user["id"],
                "type": ev.type,
                "track": track_id,
                "playback": ev.playback_id,
                "payload": json.dumps(payload),
                "occ": occurred,
            }
        )
    # one batched INSERT (executemany) instead of a round-trip per event
    await db.execute_many(
        """
        INSERT INTO events (id, user_id, type, track_id, playback_id, payload, occurred_at)
        VALUES (CAST(:id AS uuid), :u, :type, :track, :playback, :payload, CAST(:occ AS timestamptz))
        ON CONFLICT (id) DO NOTHING
        """,
        rows,
    )
    return Response(status_code=202)


@router.get("/scoring/profiles/{label}")
async def get_scoring_profile(label: str, user: dict = Depends(current_user)) -> dict:
    row = await db.fetch_one(
        "SELECT params, version FROM scoring_profiles WHERE label = :l",
        {"l": label},
    )
    return {
        "label": label,
        "params": (row or {}).get("params") or {},
        "version": (row or {}).get("version"),
    }


@router.put("/scoring/profiles/{label}")
async def put_scoring_profile(label: str, body: dict, user: dict = Depends(require_admin)) -> dict:
    params = rec_llm.validate_patch(body)
    profile = rec_llm.to_profile(params)
    await db.execute(
        """
        INSERT INTO scoring_profiles (label, params, version, updated_at)
        VALUES (:l, CAST(:p AS jsonb), 1, NOW())
        ON CONFLICT (label) DO UPDATE
        SET params = EXCLUDED.params,
            version = scoring_profiles.version + 1,
            updated_at = NOW()
        """,
        {"l": label, "p": json.dumps(profile)},
    )
    rec_mixes.clear_cache()
    return {"label": label, "params": profile}


@router.get("/config", response_model=Config)
async def get_config() -> Config:
    return Config(
        api_version="1.0",
        min_client_version="0.1.0",
        features={
            "recommendations": True,
            "lyrics": True,
            "downloads": True,
            "playlists": True,
            "offline": True,
        },
        limits={
            "maxPlaylistTracks": 5000,
            "maxUploadBytes": 100 * 1024 * 1024,
            "eventBatchSize": 100,
        },
        media={
            "supportedFormats": ["mp3", "flac", "m4a", "aac", "ogg", "opus", "wav"],
            "streamChunkSize": settings.stream_chunk_size,
        },
    )


@router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


# ── In-app updates ──────────────────────────────────────────────────────────


@router.get("/app-release/latest")
async def app_release_latest(user: dict = Depends(current_user)) -> dict:
    """Newest published client, or 404 when nothing is published yet.

    versionCode is the comparison key; the client prompts to install when its
    own BuildConfig versionCode is lower.
    """
    from ..services import releases

    rel = await releases.latest()
    if rel is None:
        raise ProblemError(404, "not_found", "No release published")
    rel.pop("_path", None)
    rel["downloadUrl"] = "/v1/app-release/apk"
    return rel


@router.get("/app-release/apk")
async def app_release_apk(user: dict = Depends(current_user)) -> Response:
    """APK bytes (single-origin: never an external URL)."""
    from fastapi.responses import FileResponse

    from ..services import releases

    rel = await releases.latest()
    if rel is None:
        raise ProblemError(404, "not_found", "No release published")
    path = await releases.apk_path_for(rel)
    if path is None:
        raise ProblemError(404, "not_found", "Release file missing on server")
    return FileResponse(
        path,
        media_type="application/vnd.android.package-archive",
        filename=f"kwhale-{rel['versionName']}.apk",
    )
