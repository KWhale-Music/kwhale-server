"""Connector base: uniform TrackMeta + async connector contract.

Adding a source = subclass Connector, implement the abstract methods,
register in connectors/__init__.py::REGISTRY. Local library is NOT a
connector — it is a track_sources row (connector='local') created by the
downloader, so search/stream code treats it uniformly.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ── quality tiers (higher = better; winner selection in dedup) ──────────────
# Scale is per *format*, not per connector: a source that can serve FLAC beats
# one serving MP3 320 regardless of the service. Refined on resolve/download.
TIER_LOCAL = 100  # downloaded file on disk
TIER_HIREZ = 95  # 24-bit FLAC (Qobuz 24, Tidal HiRes)
TIER_FLAC = 85  # 16-bit FLAC (Deezer HiFi, Qobuz, Tidal, Zvuk, Yandex Plus)
TIER_MP3_320 = 65  # MP3 320 (ICM, Deezer HQ, Zvuk, Yandex)
TIER_MP3_128 = 50  # MP3 128 (Deezer free)
TIER_AAC = 40  # AAC ~128-256 (SoundCloud HLS, YouTube Music)


@dataclass
class TrackMeta:
    """Normalised track metadata from a connector search."""

    connector: str
    external_id: str
    title: str
    artists: list[str]
    album: str = ""
    duration_sec: int = 0
    isrc: str | None = None
    cover_url: str = ""
    quality_tier: int = 0
    year: int | None = None
    track_number: int | None = None
    track_total: int | None = None
    disc_number: int | None = None
    disc_total: int | None = None
    explicit: bool | None = None
    genre: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class AlbumMeta:
    """Normalised album metadata from a connector search."""

    connector: str
    external_id: str
    title: str
    artist_name: str = ""
    year: int | None = None
    release_date: str | None = None
    cover_url: str = ""
    track_count: int | None = None
    album_type: str | None = None
    genre: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class ArtistMeta:
    """Normalised artist metadata from a connector search."""

    connector: str
    external_id: str
    name: str
    cover_url: str = ""
    genre: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)


class StreamInfo:
    """Resolved playable stream: direct URL (remote) or local file path.

    transform: connector-owned byte-stream fixup tag ('dzr_stripe',
    'dash_concat', 'hls_concat', 'vk_hls', ...). The stream service never
    interprets the tag — it asks the owning connector through the
    needs_assembly()/assemble_stream()/stream_decoder() hooks below, so all
    protocol knowledge stays inside connectors/. None = bytes pass through.
    segment_urls carries pre-parsed segments for concat-style transforms.
    """

    def __init__(
        self,
        url: str | None = None,
        file_path: Path | None = None,
        mime: str | None = None,
        transform: str | None = None,
        segment_urls: list[str] | None = None,
    ):
        if not url and not file_path:
            raise ValueError("StreamInfo needs url or file_path")
        self.url = url
        self.file_path = file_path
        self.mime = mime
        self.transform = transform
        self.segment_urls = segment_urls


@dataclass
class AccountState:
    """Result of a connect/refresh flow (secrets stay in payload_updates)."""

    ok: bool = True
    status: str = "connected"  # connected|expired|error
    message: str = ""
    profile: dict[str, Any] = field(default_factory=dict)  # non-secret display info
    payload_updates: dict[str, Any] = field(default_factory=dict)  # secrets to persist


class Connector(ABC):
    name: str = "base"

    @abstractmethod
    async def is_available(self) -> bool:
        """Can authenticate right now (has token etc)."""

    async def connect(self, payload: dict[str, str]) -> AccountState:
        """Account-based sources: validate pasted credentials, return updates+profile."""
        raise NotImplementedError(f"{self.name}: connect flow not implemented")

    async def check_account(self) -> AccountState:
        """Refresh tokens and re-read the profile; default: not implemented."""
        raise NotImplementedError(f"{self.name}: account check not implemented")

    @abstractmethod
    async def search(self, query: str, limit: int) -> list[TrackMeta]:
        """Search tracks. Returns normalised TrackMeta list."""

    @abstractmethod
    async def resolve_stream(self, external_id: str) -> StreamInfo | None:
        """Resolve a playable stream for a provider id (may be short-lived)."""

    @abstractmethod
    async def download(self, external_id: str, dest_dir: Path) -> tuple[Path, str] | None:
        """Download best quality to dest_dir. Returns (path, mime) or None."""

    async def keepalive(self) -> bool:
        """Optional periodic call to keep tokens fresh (ICM). Default: no-op ok."""
        return True

    async def album_track_number_by_source(self, external_id: str) -> int | None:
        """Optional: lazy album track-number lookup (ICM). Default: None."""
        return None

    async def search_albums(self, query: str, limit: int) -> list[AlbumMeta]:
        """Search albums. Default: none (not all providers support it)."""
        return []

    async def search_artists(self, query: str, limit: int) -> list[ArtistMeta]:
        """Search artists. Default: none (not all providers support it)."""
        return []

    async def album_tracks(self, external_id: str) -> list[TrackMeta] | None:
        """Lazy full tracklist of an online album (for album pages)."""
        return None

    async def artist_top_tracks(self, external_id: str, limit: int = 20) -> list[TrackMeta] | None:
        """Lazy artist top tracks (for artist pages)."""
        return None

    async def artist_albums(self, external_id: str, limit: int = 50) -> list[AlbumMeta] | None:
        """Lazy albums of an artist (for artist pages)."""
        return None

    async def artist_all_tracks(self, external_id: str, limit: int = 1000) -> list[TrackMeta] | None:
        """Full artist discography where the source supports pagination.

        Default falls back to the first top-tracks page. Yandex/VK override
        this with real paging so liked artists can be cached in the library.
        """
        return await self.artist_top_tracks(external_id, limit)

    async def artist_all_albums(self, external_id: str, limit: int = 200) -> list[AlbumMeta] | None:
        """Full artist album list where supported; default first page."""
        return await self.artist_albums(external_id, limit)

    # ── stream materialisation: transform handling stays connector-owned ─────

    def needs_assembly(self, info: StreamInfo) -> bool:
        """True when resolve_stream() yields a segmented stream (DASH/HLS concat
        etc.) that must be assembled into a single media file before playback."""
        return False

    async def assemble_stream(self, info: StreamInfo, dest: Path) -> None:
        """Materialise a segmented stream into dest. Implement in connectors
        whose needs_assembly() can be True; raise on failure."""
        raise NotImplementedError(f"{self.name}: assemble_stream not implemented")

    def stream_decoder(self, info: StreamInfo, url: str) -> Any | None:
        """Per-chunk decode fixup for proxied streams (e.g. on-the-fly
        decryption). Returns an object with .process(chunk, abs_offset) or None."""
        return None
