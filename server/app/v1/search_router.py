"""KWhale v1 search and suggestions."""

from __future__ import annotations

import math
from typing import Any

from fastapi import APIRouter, Depends, Query, Response

from .. import db
from ..auth import current_user
from ..services import playlists as playlists_service
from ..services import search_service
from ..services.search_rank import (
    dedupe_albums,
    dedupe_artists,
    dedupe_tracks,
    pick_top,
    sort_albums,
    sort_artists,
    sort_playlists,
    sort_tracks,
)
from .schemas import SearchHistory, SearchResponse, Suggestion, Suggestions
from .serializers import album_dto_by_ids, artist_dto_by_ids, build_playlist_dto, build_track_dtos

router = APIRouter(tags=["search"])

_ALL_TYPES = {"all", "tracks", "albums", "artists", "playlists", "mixes"}


async def _remember(user_id: int, query: str) -> None:
    q = query.strip()
    if not q:
        return
    await db.execute(
        "INSERT INTO search_history (user_id, query) VALUES (:u, :q)",
        {"u": user_id, "q": q},
    )
    await db.execute(
        """
        DELETE FROM search_history
        WHERE user_id = :u AND id NOT IN (
            SELECT id FROM search_history WHERE user_id = :u ORDER BY id DESC LIMIT 50
        )
        """,
        {"u": user_id},
    )


def _item_id(item: Any) -> int | None:
    raw = item.get("id") if isinstance(item, dict) else getattr(item, "id", None)
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def _normalise(raw: dict[str, float]) -> dict[str, float]:
    maximum = max(raw.values(), default=0.0)
    if maximum <= 0:
        return {key: 0.0 for key in raw}
    denominator = math.log1p(maximum)
    return {key: math.log1p(value) / denominator for key, value in raw.items()}


async def _top_popularity(tracks: list[Any], albums: list[Any], artists: list[Any]) -> dict[str, float]:
    """Cheap, local popularity signals for the handful of top candidates.

    This is not a global ranking service: it only decides between an exact
    artist name and an exact album title (e.g. 'Nevermind' vs a band with that
    name) using local plays, likes and library presence.
    """
    out: dict[str, float] = {}

    artist_ids = [i for i in (_item_id(a) for a in artists[:5]) if i]
    if artist_ids:
        params = {f"a{n}": value for n, value in enumerate(artist_ids)}
        rows = await db.fetch_all(
            f"""
            SELECT ar.id,
                   (SELECT COUNT(*) FROM albums al WHERE al.artist_id = ar.id) AS albums,
                   (SELECT COUNT(*) FROM track_artists ta
                    JOIN track_sources s ON s.track_id = ta.track_id
                                        AND s.connector = 'local'
                    WHERE ta.artist_id = ar.id) AS local_tracks
            FROM artists ar
            WHERE ar.id IN ({", ".join(":" + key for key in params)})
            """,
            params,
        )
        raw = {
            f"artist:{r['id']}": 0.6 * float(r["albums"] or 0) + 0.4 * float(r["local_tracks"] or 0)
            for r in rows
        }
        out.update(_normalise(raw))

    album_ids = [i for i in (_item_id(a) for a in albums[:5]) if i]
    if album_ids:
        params = {f"a{n}": value for n, value in enumerate(album_ids)}
        rows = await db.fetch_all(
            f"""
            SELECT al.id, COALESCE(al.track_count, 0) AS track_count,
                   (SELECT COUNT(*) FROM favorite_albums fa WHERE fa.album_id = al.id) AS favs,
                   (SELECT COUNT(*) FROM tracks t
                    JOIN track_sources s ON s.track_id = t.id AND s.connector = 'local'
                    JOIN track_artists ta ON ta.track_id = t.id AND ta.artist_id = ar.id
                    WHERE lower(regexp_replace(t.album, '\\s+', '', 'g'))
                        = lower(regexp_replace(al.title, '\\s+', '', 'g'))) AS local_tracks
            FROM albums al
            JOIN artists ar ON ar.id = al.artist_id
            WHERE al.id IN ({", ".join(":" + key for key in params)})
            """,
            params,
        )
        raw = {
            f"album:{r['id']}": (
                0.4 * float(r["track_count"] or 0)
                + 0.3 * float(r["favs"] or 0)
                + 0.3 * float(r["local_tracks"] or 0)
            )
            for r in rows
        }
        out.update(_normalise(raw))

    track_ids = [i for i in (_item_id(t) for t in tracks[:8]) if i]
    if track_ids:
        params = {f"t{n}": value for n, value in enumerate(track_ids)}
        rows = await db.fetch_all(
            f"""
            SELECT t.id,
                   (SELECT COUNT(*) FROM favorites f WHERE f.track_id = t.id) AS likes,
                   (SELECT COUNT(*) FROM events e
                    WHERE e.track_id = t.id AND e.type = 'play') AS plays,
                   (SELECT COUNT(*) FROM track_sources s
                    WHERE s.track_id = t.id AND s.connector = 'local') AS local
            FROM tracks t
            WHERE t.id IN ({", ".join(":" + key for key in params)})
            """,
            params,
        )
        raw = {
            f"track:{r['id']}": (
                0.5 * float(r["plays"] or 0)
                + 0.3 * float(r["likes"] or 0)
                + 0.2 * float(r["local"] or 0)
            )
            for r in rows
        }
        out.update(_normalise(raw))

    return out


@router.get("/search", response_model=SearchResponse)
async def search(
    q: str = Query(..., min_length=1, max_length=200),
    type: str = Query("all"),
    limit: int | None = Query(None, ge=1, le=100),
    wait: bool = Query(False),
    user: dict = Depends(current_user),
) -> SearchResponse:
    section = type.strip().lower()
    if section not in _ALL_TYPES:
        from ..errors import ProblemError

        raise ProblemError(400, "validation_failed", f"unknown search type: {type}")
    lim = limit or 50
    if not wait:
        await _remember(user["id"], q)

    local_ids = await search_service.search_local_ids(q, lim)
    remote: dict[str, Any] = (
        await search_service.search(q, lim, user["id"], wait_all=wait, with_dtos=False)
        if section in ("all", "tracks", "albums", "artists")
        else {"tracks": {}, "albums": {}, "artists": {}, "partial": False}
    )

    remote_ids = [int(item["id"]) for item in (remote.get("tracks") or {}).get("items") or []]
    ordered: list[int] = []
    seen: set[int] = set()
    for tid in [*local_ids, *remote_ids]:
        if tid not in seen:
            seen.add(tid)
            ordered.append(tid)
    track_ids = ordered[:lim] if section in ("all", "tracks") else []
    track_items = sort_tracks(
        q, dedupe_tracks(await build_track_dtos(track_ids, user["id"]))
    )

    album_ids = [int(a["id"]) for a in (remote.get("albums") or {}).get("items") or []]
    album_items = (
        sort_albums(q, dedupe_albums(await album_dto_by_ids(list(dict.fromkeys(album_ids)), user["id"])))
        if section in ("all", "albums")
        else []
    )

    artist_ids = [int(a["id"]) for a in (remote.get("artists") or {}).get("items") or []]
    artist_items = (
        sort_artists(
            q,
            dedupe_artists(
                await artist_dto_by_ids(list(dict.fromkeys(artist_ids)), user["id"], hide_empty=True)
            ),
        )
        if section in ("all", "artists")
        else []
    )

    playlist_items = []
    if section in ("all", "playlists"):
        rows = await playlists_service.search_playlists(q, user["id"], lim)
        playlist_items = sort_playlists(
            q, [await build_playlist_dto(r, user["id"]) for r in rows]
        )

    popularity = await _top_popularity(track_items, album_items, artist_items)
    top = pick_top(
        q, track_items, album_items, artist_items, playlist_items, popularity
    )
    total = len(track_items) + len(album_items) + len(artist_items) + len(playlist_items)
    return SearchResponse(
        query=q,
        tracks=track_items,
        albums=album_items,
        artists=artist_items,
        playlists=playlist_items,
        mixes=[],
        top=top,
        total=total,
        partial=bool(remote.get("partial")),
    )


@router.get("/search/suggestions", response_model=Suggestions)
async def suggestions(
    q: str = Query(..., min_length=1, max_length=100),
    user: dict = Depends(current_user),
) -> Suggestions:
    needle = f"%{q.strip()}%"
    rows = await db.fetch_all(
        """
        SELECT name AS text, 'artist' AS kind FROM artists WHERE name ILIKE :q
        UNION ALL
        SELECT title AS text, 'track' AS kind FROM tracks WHERE title ILIKE :q
        UNION ALL
        SELECT title AS text, 'album' AS kind FROM albums WHERE title ILIKE :q
        LIMIT 20
        """,
        {"q": needle},
    )
    seen: set[tuple[str, str]] = set()
    items: list[Suggestion] = []
    for r in rows:
        key = (r["text"].lower(), r["kind"])
        if key in seen:
            continue
        seen.add(key)
        items.append(Suggestion(text=r["text"], type=r["kind"]))
        if len(items) >= 10:
            break
    if not items:
        items = [Suggestion(text=q, type="query")]
    return Suggestions(items=items)


@router.get("/search/history", response_model=SearchHistory)
async def get_search_history(user: dict = Depends(current_user)) -> SearchHistory:
    rows = await db.fetch_all(
        "SELECT query FROM search_history WHERE user_id = :u ORDER BY id DESC LIMIT 50",
        {"u": user["id"]},
    )
    return SearchHistory(items=[r["query"] for r in rows])


@router.delete("/search/history", status_code=204)
async def clear_search_history(user: dict = Depends(current_user)) -> Response:
    await db.execute("DELETE FROM search_history WHERE user_id = :u", {"u": user["id"]})
    return Response(status_code=204)
