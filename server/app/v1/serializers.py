"""DB rows -> canonical KWhale v1 DTOs.

This is the only public serialization layer. Internal services keep their
existing row shapes; the v1 API exposes contract models directly.
"""

from __future__ import annotations

import re
from typing import Any, Literal

from .. import db
from ..services import dto as old_dto
from . import images
from .schemas import (
    Album,
    AlbumRef,
    Artist,
    ArtistRef,
    Genre,
    ImageRef,
    Lyrics,
    LyricsLine,
    Playlist,
    PlaylistPermissions,
    Track,
    UserRef,
)


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-") or "unknown"


def _iso(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return value.isoformat()


def genre_from_name(name: str | None) -> Genre | None:
    if not name:
        return None
    return Genre(id=_slug(name), name=name)


def user_ref(row: dict[str, Any] | None, fallback_id: int | None = None) -> UserRef:
    if row is None:
        return UserRef(
            id=str(fallback_id or 0),
            display_name="Unknown",
            avatar=images.placeholder_ref(),
        )
    return UserRef(
        id=str(row["id"]),
        display_name=row.get("display_name") or row.get("username") or "Unknown",
        avatar=images.placeholder_ref(),
    )


async def _album_covers(album_ids: list[int | None]) -> dict[int, str]:
    ids = [int(i) for i in album_ids if i is not None]
    if not ids:
        return {}
    placeholders = [f":a{i}" for i in range(len(ids))]
    rows = await db.fetch_all(
        f"SELECT id, cover_url FROM albums WHERE id IN ({', '.join(placeholders)})",
        {f"a{i}": v for i, v in enumerate(ids)},
    )
    return {r["id"]: r["cover_url"] for r in rows if r.get("cover_url")}


async def _censorship_states(track_ids: list[int], user_id: int | None) -> set[int]:
    """Track ids the user has flagged for censorship review (open reports)."""
    if not user_id or not track_ids:
        return set()
    placeholders = [f":t{i}" for i in range(len(track_ids))]
    params = {f"t{i}": tid for i, tid in enumerate(track_ids)}
    params["u"] = user_id
    rows = await db.fetch_all(
        f"SELECT track_id FROM censorship_reports "
        f"WHERE user_id = :u AND track_id IN ({', '.join(placeholders)})",
        params,
    )
    return {int(r["track_id"]) for r in rows}


async def _download_states(track_ids: list[int]) -> dict[int, tuple[bool, str]]:
    if not track_ids:
        return {}
    placeholders = [f":d{i}" for i in range(len(track_ids))]
    rows = await db.fetch_all(
        f"SELECT track_id, status FROM download_jobs WHERE track_id IN ({', '.join(placeholders)})",
        {f"d{i}": v for i, v in enumerate(track_ids)},
    )
    return {
        r["track_id"]: (
            r["status"] == "done",
            "downloading"
            if r["status"] == "running"
            else ("pending" if r["status"] == "pending" else ("failed" if r["status"] == "failed" else "pending")),
        )
        for r in rows
    }


async def build_track_dtos(track_ids: list[int], user_id: int | None) -> list[Track]:
    rows = await old_dto.build_track_dtos(track_ids, user_id)
    if not rows:
        return []
    states = await _download_states(track_ids)
    censored = await _censorship_states(track_ids, user_id)
    covers = await _album_covers([r.get("album_id") for r in rows])
    all_urls: list[str | None] = []
    for r in rows:
        all_urls.append(r.get("cover_url"))
        if r.get("album_id"):
            all_urls.append(covers.get(r["album_id"]) or r.get("cover_url"))
    image_map = await images.image_refs_batch(all_urls)
    out: list[Track] = []
    for r in rows:
        track_url = (r.get("cover_url") or "").strip()
        track_cover = image_map.get(track_url) or images.placeholder_ref()
        album = None
        if r.get("album_id") and r.get("album"):
            album_url = (covers.get(r["album_id"]) or r.get("cover_url") or "").strip()
            album = AlbumRef(
                id=str(r["album_id"]),
                title=r["album"],
                cover=image_map.get(album_url) or images.placeholder_ref(),
            )
        artists: list[ArtistRef] = []
        artist_ids = r.get("artist_ids") or []
        for idx, name in enumerate(r.get("artists") or []):
            aid = artist_ids[idx] if idx < len(artist_ids) else None
            artists.append(ArtistRef(id=str(aid) if aid is not None else f"unknown-{_slug(name)}", name=name))
        is_local = any(s.get("connector") == "local" for s in (r.get("sources") or []))
        _job_done, job_status = states.get(r["id"], (False, "pending"))
        download_status: Literal["pending", "downloading", "ready", "failed"]
        if is_local:
            download_status = "ready"
        elif job_status == "failed":
            download_status = "failed"
        elif job_status == "downloading":
            download_status = "downloading"
        else:
            download_status = "pending"
        out.append(
            Track(
                id=str(r["id"]),
                title=r["title"],
                artists=artists,
                album=album,
                duration_ms=int((r.get("duration_sec") or 0) * 1000),
                explicit=bool(r.get("explicit")),
                cover=track_cover,
                genre=genre_from_name(r.get("genre")),
                year=r.get("year"),
                track_number=r.get("track_number"),
                disc_number=r.get("disc_number"),
                is_liked=bool(r.get("is_favorite")),
                is_disliked=bool(r.get("is_disliked")),
                has_lyrics=bool(r.get("has_lyrics")),
                is_downloaded=is_local,
                download_status=download_status,
                is_censorship_reported=int(r["id"]) in censored,
                preview_available=False,
                play_count=None,
                created_at=None,
            )
        )
    return out


async def track_dto(track_id: int, user_id: int | None) -> Track | None:
    items = await build_track_dtos([track_id], user_id)
    return items[0] if items else None


async def build_album_dto(
    row: dict[str, Any],
    *,
    tracks: list[Track] | None = None,
    cover_ref: ImageRef | None = None,
) -> Album:
    artist = row.get("artist") or {}
    artist_id = artist.get("id") if isinstance(artist, dict) else None
    artist_name = artist.get("name") if isinstance(artist, dict) else None
    return Album(
        id=str(row["id"]),
        title=row["title"],
        artists=[
            ArtistRef(
                id=str(artist_id) if artist_id is not None else f"unknown-{_slug(artist_name or '')}",
                name=artist_name or "Unknown",
            )
        ],
        cover=cover_ref or await images.image_ref(row.get("cover_url")),
        year=row.get("year"),
        release_date=row.get("release_date"),
        track_count=row.get("track_count") or (len(tracks) if tracks else 0),
        type=row.get("album_type") or row.get("type"),
        explicit=bool(row.get("explicit")),
        is_favorite=bool(row.get("is_favorite")),
        tracks=tracks or [],
        created_at=None,
        added_at=row.get("added_at"),
    )


async def album_dto_by_ids(
    album_ids: list[int], user_id: int | None = None
) -> list[Album]:
    from ..services.albums import album_dto_by_ids as old_album_dtos

    rows = await old_album_dtos(album_ids, user_id)
    image_map = await images.image_refs_batch([r.get("cover_url") for r in rows])
    return [
        await build_album_dto(
            r, cover_ref=image_map.get((r.get("cover_url") or "").strip()) or images.placeholder_ref()
        )
        for r in rows
    ]


async def build_artist_dto(
    row: dict[str, Any],
    *,
    albums: list[Album] | None = None,
    top_tracks: list[Track] | None = None,
    latest_release: Album | None = None,
    avatar_ref: ImageRef | None = None,
) -> Artist:
    genre = row.get("genre")
    genre_item = genre_from_name(genre)
    album_count = len(albums) if albums is not None else int(row.get("album_count") or 0)
    # Artists without their own artwork borrow it from the latest release or
    # the first album — an artist page must never render as a grey placeholder
    # while its releases have covers.
    avatar = avatar_ref or await images.image_ref(row.get("cover_url"))
    if avatar.id == images.PLACEHOLDER_ID:
        for candidate in (
            (latest_release.cover if latest_release else None),
            (albums[0].cover if albums else None),
        ):
            if candidate is not None and candidate.id != images.PLACEHOLDER_ID:
                avatar = candidate
                break
    return Artist(
        id=str(row["id"]),
        name=row["name"],
        avatar=avatar,
        bio=None,
        genres=[genre_item] if genre_item is not None else [],
        followers=0,
        album_count=album_count,
        is_favorite=bool(row.get("is_favorite")),
        latest_release=latest_release,
        top_tracks=top_tracks or [],
        albums=albums or [],
    )


async def artist_dto_by_ids(
    artist_ids: list[int], user_id: int | None = None, hide_empty: bool = False
) -> list[Artist]:
    from ..services.artists import artist_dto_by_ids as old_artist_dtos

    rows = await old_artist_dtos(artist_ids, user_id, hide_empty=hide_empty)
    image_map = await images.image_refs_batch([r.get("cover_url") for r in rows])
    return [
        await build_artist_dto(
            r, avatar_ref=image_map.get((r.get("cover_url") or "").strip()) or images.placeholder_ref()
        )
        for r in rows
    ]


async def _owner(user_id: int) -> dict[str, Any] | None:
    return await db.fetch_one("SELECT id, username, display_name FROM users WHERE id = :i", {"i": user_id})


async def build_playlist_dto(
    row: dict[str, Any],
    user_id: int,
    *,
    tracks: list[Track] | None = None,
) -> Playlist:
    owner_row = await _owner(row["user_id"])
    is_owner = row["user_id"] == user_id
    return Playlist(
        id=str(row["id"]),
        title=row["name"],
        description=row.get("description"),
        owner=user_ref(owner_row, row["user_id"]),
        cover=(
            ImageRef(id=str(row["cover_image_id"]))
            if row.get("cover_image_id")
            else images.placeholder_ref()
        ),
        track_count=row.get("track_count") or (len(tracks) if tracks is not None else 0),
        visibility=row.get("visibility") or "private",
        is_system=bool(row.get("is_system")),
        is_followed=bool(row.get("is_following")),
        permissions=PlaylistPermissions(
            can_edit=is_owner,
            can_delete=is_owner,
            can_add_tracks=is_owner,
        ),
        tracks=tracks or [],
        created_at=_iso(row.get("created_at")),
        updated_at=_iso(row.get("updated_at")),
    )


async def playlist_page(rows: list[dict[str, Any]], user_id: int) -> list[Playlist]:
    return [await build_playlist_dto(r, user_id) for r in rows]


def parse_lrc(lrc: str | None) -> list[LyricsLine]:
    if not lrc:
        return []
    out: list[LyricsLine] = []
    for raw in lrc.splitlines():
        match = re.match(r"\[(\d+):(\d+(?:\.\d+)?)\](.*)", raw.strip())
        if not match:
            continue
        minutes, seconds, text = match.groups()
        start_ms = int((int(minutes) * 60 + float(seconds)) * 1000)
        out.append(LyricsLine(start_ms=start_ms, text=text.strip()))
    return out


def build_lyrics(track_id: int, got: dict[str, Any] | None) -> Lyrics:
    got = got or {}
    plain = got.get("lyrics")
    lrc = got.get("lrc")
    synced = parse_lrc(lrc)
    return Lyrics(
        track_id=str(track_id),
        provider="internal",
        language="ru",
        plain=plain,
        synced=synced,
        has_timestamps=bool(synced),
    )
