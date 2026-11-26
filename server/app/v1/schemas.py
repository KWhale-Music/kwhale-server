"""Canonical KWhale Core v1 schemas (camelCase on the wire)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel


class ApiModel(BaseModel):
    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
        from_attributes=True,
        extra="ignore",
    )


class Problem(ApiModel):
    type: str
    title: str
    status: int
    detail: str | None = None
    instance: str | None = None
    code: str | None = None
    fields: dict[str, str] = Field(default_factory=dict)


class ImageRef(ApiModel):
    id: str
    width: int | None = None
    height: int | None = None
    blur_hash: str | None = None


class User(ApiModel):
    id: str
    display_name: str
    email: str | None = None
    avatar: ImageRef
    created_at: str | None = None


class UserRef(ApiModel):
    id: str
    display_name: str
    avatar: ImageRef


class UserUpdate(ApiModel):
    display_name: str | None = None


class AuthResponse(ApiModel):
    access_token: str
    refresh_token: str
    expires_in: int
    token_type: str = "Bearer"
    user: User


class RegisterRequest(ApiModel):
    email: str
    password: str
    display_name: str
    invite_token: str
    device_id: str | None = None


class LoginRequest(ApiModel):
    email: str
    password: str
    device_id: str | None = None


class RefreshRequest(ApiModel):
    refresh_token: str


class LogoutRequest(ApiModel):
    refresh_token: str


class ArtistRef(ApiModel):
    id: str
    name: str


class AlbumRef(ApiModel):
    id: str
    title: str
    cover: ImageRef


class Genre(ApiModel):
    id: str
    name: str
    image: ImageRef | None = None


DownloadStatus = Literal["pending", "downloading", "ready", "failed"]


class Track(ApiModel):
    id: str
    title: str
    artists: list[ArtistRef]
    album: AlbumRef | None = None
    duration_ms: int
    explicit: bool = False
    cover: ImageRef
    genre: Genre | None = None
    year: int | None = None
    track_number: int | None = None
    disc_number: int | None = None
    is_liked: bool = False
    is_disliked: bool = False
    has_lyrics: bool = False
    is_downloaded: bool = False
    download_status: DownloadStatus = "pending"
    is_censorship_reported: bool = False
    preview_available: bool = False
    play_count: int | None = None
    created_at: str | None = None


class TrackQualityInfo(ApiModel):
    codec: str | None = None
    bitrate_kbps: int | None = None
    sample_rate: int | None = None
    bit_depth: int | None = None
    channels: int | None = None
    lossless: bool = False
    tier: int = 0
    bytes: int | None = None
    measured_duration_ms: int | None = None
    error: str | None = None


class TrackSourceInfo(ApiModel):
    connector: str
    available: bool = True
    quality_tier: int | None = None


class TrackInfo(ApiModel):
    """Everything the player's «О треке» sheet shows."""

    track: Track
    isrc: str | None = None
    quality: TrackQualityInfo | None = None
    sources: list[TrackSourceInfo] = []


class CensorshipReportRequest(ApiModel):
    reason: str = "censored"
    comment: str | None = None
    position_ms: int | None = None


class CensorshipReportResponse(ApiModel):
    ok: bool = True
    report_id: str = ""
    status: str = "pending"


class Album(ApiModel):
    id: str
    title: str
    artists: list[ArtistRef]
    cover: ImageRef
    year: int | None = None
    release_date: str | None = None
    track_count: int = 0
    type: str | None = None
    explicit: bool = False
    is_favorite: bool = False
    tracks: list[Track] = Field(default_factory=list)
    created_at: str | None = None
    added_at: str | None = None


class Artist(ApiModel):
    id: str
    name: str
    avatar: ImageRef
    bio: str | None = None
    genres: list[Genre] = Field(default_factory=list)
    followers: int = 0
    album_count: int = 0
    is_favorite: bool = False
    latest_release: Album | None = None
    top_tracks: list[Track] = Field(default_factory=list)
    albums: list[Album] = Field(default_factory=list)


class PlaylistPermissions(ApiModel):
    can_edit: bool = False
    can_delete: bool = False
    can_add_tracks: bool = False


class Playlist(ApiModel):
    id: str
    title: str
    description: str | None = None
    owner: UserRef
    cover: ImageRef
    track_count: int = 0
    visibility: Literal["private", "unlisted", "public"] = "private"
    is_system: bool = False
    is_followed: bool = False
    permissions: PlaylistPermissions
    tracks: list[Track] = Field(default_factory=list)
    created_at: str | None = None
    updated_at: str | None = None


class PlaylistCreate(ApiModel):
    title: str
    description: str | None = None
    visibility: Literal["private", "unlisted", "public"] = "private"


class PlaylistUpdate(ApiModel):
    title: str | None = None
    description: str | None = None
    visibility: Literal["private", "unlisted", "public"] | None = None


class PlaylistTracksAdd(ApiModel):
    track_ids: list[str] = Field(min_length=1, max_length=1000)
    position: int | None = None


class PlaylistReorder(ApiModel):
    track_ids: list[str] = Field(min_length=1, max_length=5000)


class LikeAllResult(ApiModel):
    requested: int
    liked: int
    already_liked: int
    queued: int
    failed: int


class TracksPage(ApiModel):
    items: list[Track] = Field(default_factory=list)
    next_cursor: str | None = None
    total: int | None = None


class AlbumsPage(ApiModel):
    items: list[Album] = Field(default_factory=list)
    next_cursor: str | None = None


class ArtistsPage(ApiModel):
    items: list[Artist] = Field(default_factory=list)
    next_cursor: str | None = None


class PlaylistsPage(ApiModel):
    items: list[Playlist] = Field(default_factory=list)
    next_cursor: str | None = None
    total: int | None = None


class TrackRecommendation(ApiModel):
    track: Track
    reason: str | None = None
    reason_code: str | None = None
    score: float | None = None


class TrackRecommendationsPage(ApiModel):
    items: list[TrackRecommendation] = Field(default_factory=list)
    next_cursor: str | None = None


class MixesPage(ApiModel):
    items: list[Mix] = Field(default_factory=list)
    next_cursor: str | None = None


class PromptMixRequest(ApiModel):
    prompt: str
    size: int = 40
    # Optional per-request LLM overrides (OpenAI-compatible endpoint); fall back to
    # the env-configured DeepSeek defaults when absent.
    llm_api_key: str | None = None
    llm_endpoint: str | None = None
    llm_model: str | None = None


class GenresResponse(ApiModel):
    items: list[Genre] = Field(default_factory=list)


class HistoryEntry(ApiModel):
    track: Track
    played_at: str
    context: str | None = None


class HistoryPage(ApiModel):
    items: list[HistoryEntry] = Field(default_factory=list)
    next_cursor: str | None = None
    total: int | None = None


class SearchTop(ApiModel):
    type: Literal["track", "album", "artist", "playlist"]
    track: Track | None = None
    album: Album | None = None
    artist: Artist | None = None
    playlist: Playlist | None = None


class SearchResponse(ApiModel):
    query: str
    tracks: list[Track] = Field(default_factory=list)
    albums: list[Album] = Field(default_factory=list)
    artists: list[Artist] = Field(default_factory=list)
    playlists: list[Playlist] = Field(default_factory=list)
    mixes: list[dict[str, Any]] = Field(default_factory=list)
    top: SearchTop | None = None
    total: int = 0
    partial: bool = False


class Suggestion(ApiModel):
    text: str
    type: Literal["query", "artist", "track", "album", "playlist"] = "query"


class Suggestions(ApiModel):
    items: list[Suggestion] = Field(default_factory=list)


class SearchHistory(ApiModel):
    items: list[str] = Field(default_factory=list)


class LyricsLine(ApiModel):
    start_ms: int
    text: str
    end_ms: int | None = None


class Lyrics(ApiModel):
    track_id: str
    provider: str = "internal"
    language: str = "ru"
    plain: str | None = None
    synced: list[LyricsLine] = Field(default_factory=list)
    has_timestamps: bool = False


class SeedRef(ApiModel):
    type: Literal["track", "artist", "album", "playlist", "genre"]
    id: str


class Mix(ApiModel):
    id: str
    title: str
    subtitle: str | None = None
    cover: ImageRef | None = None
    kind: str = "personal"
    seed: SeedRef | None = None
    track_count: int = 0
    is_followed: bool = False
    created_at: str | None = None
    updated_at: str | None = None


class Onboarding(ApiModel):
    genres: list[Genre] = Field(default_factory=list)
    artists: list[ArtistRef] = Field(default_factory=list)


class Banner(ApiModel):
    id: str
    title: str
    subtitle: str | None = None
    image: ImageRef
    action: str | None = None


class HomeAction(ApiModel):
    id: str
    title: str
    subtitle: str | None = None
    icon: str | None = None
    action: str


class HomeItem(ApiModel):
    type: Literal["track", "album", "artist", "playlist", "mix", "banner", "action"]
    track: Track | None = None
    album: Album | None = None
    artist: Artist | None = None
    playlist: Playlist | None = None
    mix: Mix | None = None
    banner: Banner | None = None
    action: HomeAction | None = None


class HomeSection(ApiModel):
    id: str
    kind: Literal[
        "banner", "actions", "shelf", "list", "recommendation", "mix", "chart", "new-release"
    ]
    layout: Literal["carousel", "grid", "list", "banner", "actions"] = "list"
    title: str | None = None
    subtitle: str | None = None
    items: list[HomeItem] = Field(default_factory=list)
    see_all: bool = False
    next_cursor: str | None = None


class HomeResponse(ApiModel):
    sections: list[HomeSection] = Field(default_factory=list)


class EventContext(ApiModel):
    source: str | None = None
    mix_id: str | None = None
    playlist_id: str | None = None


EventType = Literal[
    "play",
    "pause",
    "resume",
    "seek",
    "complete",
    "skip",
    "repeat",
    "like",
    "unlike",
    "dislike",
    "undislike",
    "impression",
    "click",
    "add_to_playlist",
    "remove_from_playlist",
    "download",
    "playlist_add",
    "share",
    "search",
]

EventSurface = Literal[
    "home",
    "search",
    "library",
    "playlist",
    "album",
    "artist",
    "mix",
    "player",
    "recommendation",
    "notification",
    "external",
]


class Event(ApiModel):
    id: str
    type: EventType
    track_id: str | None = None
    playback_id: str | None = None
    context: EventContext | None = None
    position_ms: int | None = None
    listened_ms: int | None = None
    track_duration_ms: int | None = None
    started_at: str | None = None
    ended_at: str | None = None
    timezone_offset_minutes: int | None = None
    surface: EventSurface | None = None
    session_id: str | None = None
    device_id: str | None = None
    query: str | None = None
    result_count: int | None = None
    clicked_position: int | None = None
    occurred_at: str


class EventBatch(ApiModel):
    events: list[Event] = Field(min_length=1, max_length=100)


class Config(ApiModel):
    api_version: str = "1.0"
    min_client_version: str = "0.1.0"
    features: dict[str, bool] = Field(default_factory=dict)
    limits: dict[str, int] = Field(default_factory=dict)
    media: dict[str, Any] = Field(default_factory=dict)


ShareKind = Literal["track", "album", "artist"]


class SharePlatformLink(ApiModel):
    platform: str
    label: str
    url: str
    go_url: str
    exact: bool = False


class ShareCreateRequest(ApiModel):
    kind: ShareKind = "track"
    id: str = Field(min_length=1, max_length=64)


class ShareLink(ApiModel):
    id: str
    kind: ShareKind
    entity_id: str
    title: str
    subtitle: str | None = None
    meta: str | None = None
    image_id: str
    share_url: str
    card_url: str
    cover_url: str
    deep_link: str
    expires_at: str
    ttl_days: int
    clicks_total: int = 0
    platforms: list[SharePlatformLink] = Field(default_factory=list)
