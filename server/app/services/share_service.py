"""Share links service: canonical short links with sliding TTL for tracks, albums, and artists."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import secrets
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Literal
from urllib.parse import quote, quote_plus, urlparse

from fastapi import Request

from .. import db
from ..config import settings
from ..errors import ProblemError
from ..v1 import images


def request_origin(request: Request) -> str:
    """Resolve external origin cleanly across Prague Edge, Caddy, and LAN direct."""
    raw_proto = (
        request.headers.get("x-forwarded-proto") or request.url.scheme or "https"
    ).split(",")[0].strip()
    proto = raw_proto if raw_proto in ("http", "https") else "https"
    raw_host = (
        request.headers.get("x-forwarded-host")
        or request.headers.get("host")
        or request.url.netloc
        or "localhost:19001"
    )
    host = raw_host.split(",")[0].strip()
    return f"{proto}://{host}".rstrip("/")

log = logging.getLogger("kwhale.share")

ShareKind = Literal["track", "album", "artist"]

_SHARE_ALPHABET = "abcdefghijklmnopqrstuvwxyz0123456789"
_SHARE_ID_RE = re.compile(r"^[A-Za-z0-9_-]{4,32}$")
_DIGITS_RE = re.compile(r"^\d+$")
_YTM_TRACK_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")
_YTM_ALBUM_BROWSE_RE = re.compile(r"^(?:MPRE|VL|PL)[A-Za-z0-9_-]+$")
_YTM_ALBUM_LIST_RE = re.compile(r"^OLAK[A-Za-z0-9_-]+$")
_YTM_ARTIST_RE = re.compile(r"^UC[A-Za-z0-9_-]{20,}$")
_VK_OWNER_PAIR_RE = re.compile(r"^-?\d+_\d+(?:_[A-Za-z0-9]+)?$")
_VK_ARTIST_SLUG_RE = re.compile(r"^[A-Za-z0-9._-]{2,}$")

# Ordered display preference across streaming services.
PLATFORM_ORDER: tuple[str, ...] = (
    "yandex",
    "apple",
    "ytmusic",
    "soundcloud",
    "vk",
    "deezer",
    "tidal",
    "qobuz",
    "zvuk",
)

PLATFORM_LABELS: dict[str, str] = {
    "yandex": "Яндекс Музыка",
    "apple": "Apple Music",
    "ytmusic": "YouTube Music",
    "soundcloud": "SoundCloud",
    "vk": "VK Музыка",
    "deezer": "Deezer",
    "tidal": "Tidal",
    "qobuz": "Qobuz",
    "zvuk": "Звук",
}

CONNECTOR_TO_PLATFORM: dict[str, str] = {
    "icm": "apple",
    "yandex": "yandex",
    "ytmusic": "ytmusic",
    "soundcloud": "soundcloud",
    "vk": "vk",
    "deezer": "deezer",
    "tidal": "tidal",
    "qobuz": "qobuz",
    "zvuk": "zvuk",
}

PLATFORM_HOSTS: dict[str, frozenset[str]] = {
    "yandex": frozenset({"music.yandex.ru", "music.yandex.com"}),
    "apple": frozenset({"music.apple.com", "itunes.apple.com"}),
    "ytmusic": frozenset({"music.youtube.com", "www.youtube.com", "youtube.com", "youtu.be"}),
    "soundcloud": frozenset({"soundcloud.com", "www.soundcloud.com", "m.soundcloud.com"}),
    "vk": frozenset({"vk.com", "www.vk.com", "m.vk.com"}),
    "deezer": frozenset({"www.deezer.com", "deezer.com", "link.deezer.com"}),
    "tidal": frozenset({"tidal.com", "listen.tidal.com"}),
    "qobuz": frozenset({"play.qobuz.com", "open.qobuz.com", "www.qobuz.com"}),
    "zvuk": frozenset({"zvuk.com", "www.zvuk.com"}),
}


class ShareNotFoundError(Exception):
    """Raised when a share id does not exist or is malformed."""


class ShareExpiredError(Exception):
    """Raised when a share link exists in DB but its TTL has expired."""

    def __init__(self, record: ShareRecord) -> None:
        super().__init__(f"share {record.id} expired")
        self.record = record


@dataclass(slots=True)
class ShareEntity:
    kind: ShareKind
    entity_id: int
    title: str
    subtitle: str | None
    meta: str | None
    duration_sec: int | None
    image_id: str
    sources: dict[str, str]


@dataclass(slots=True)
class PlatformLink:
    platform: str
    label: str
    url: str
    exact: bool


@dataclass(slots=True)
class ShareRecord:
    id: str
    kind: ShareKind
    entity_id: int
    title: str
    subtitle: str | None
    meta: str | None
    duration_sec: int | None
    image_id: str
    platforms: list[PlatformLink]
    clicks_total: int
    created_at: datetime
    updated_at: datetime
    expires_at: datetime

    @property
    def is_expired(self) -> bool:
        return self.expires_at <= datetime.now(UTC)


def format_duration(seconds: int | None) -> str | None:
    if not seconds or seconds <= 0:
        return None
    hrs, rem = divmod(int(seconds), 3600)
    mins, secs = divmod(rem, 60)
    if hrs:
        return f"{hrs}:{mins:02d}:{secs:02d}"
    return f"{mins}:{secs:02d}"


def _ru_plural(n: int, one: str, few: str, many: str) -> str:
    n_abs = abs(n)
    mod100 = n_abs % 100
    mod10 = n_abs % 10
    if 11 <= mod100 <= 14:
        return many
    if mod10 == 1:
        return one
    if 2 <= mod10 <= 4:
        return few
    return many


def is_safe_redirect_url(platform: str, url: str) -> bool:
    """Validate destination host against explicit platform allowlist before 302."""
    hosts = PLATFORM_HOSTS.get(platform)
    if not hosts or not url:
        return False
    parsed = urlparse(url)
    return parsed.scheme == "https" and (parsed.hostname or "").lower() in hosts


def deep_link_for(kind: ShareKind, entity_id: int) -> str:
    return f"kwhale://{kind}/{entity_id}"


def android_intent_uri(kind: ShareKind, entity_id: int, fallback_url: str) -> str:
    encoded_fallback = quote(fallback_url, safe="")
    return (
        f"intent://{kind}/{entity_id}"
        f"#Intent;scheme=kwhale;package=org.dueattendant149.kwhale;"
        f"S.browser_fallback_url={encoded_fallback};end"
    )


def _new_share_id() -> str:
    return "".join(secrets.choice(_SHARE_ALPHABET) for _ in range(8))


def _is_synthetic_id(external_id: str) -> bool:
    low = external_id.lower()
    return (
        not external_id
        or low.startswith("local::")
        or low.startswith("custom_")
        or low.startswith("album:")
        or low.startswith("track:")
    )


async def _connected_platforms() -> set[str]:
    rows = await db.fetch_all(
        """
        SELECT connector
        FROM source_accounts
        WHERE enabled = TRUE AND status = 'connected'
        """
    )
    out: set[str] = set()
    for r in rows:
        plat = CONNECTOR_TO_PLATFORM.get(str(r["connector"]).lower())
        if plat:
            out.add(plat)
    return out


async def _load_entity(kind: ShareKind, entity_id: int) -> ShareEntity | None:
    if kind == "track":
        row = await db.fetch_one(
            """
            SELECT id, title, artists, album, duration_sec, cover_url
            FROM tracks
            WHERE id = :i
            """,
            {"i": entity_id},
        )
        if row is None:
            return None
        artists = [str(a).strip() for a in (row.get("artists") or []) if str(a).strip()]
        subtitle = ", ".join(artists) or None
        meta_parts: list[str] = []
        if row.get("album"):
            meta_parts.append(str(row["album"]).strip())
        dur_str = format_duration(row.get("duration_sec"))
        if dur_str:
            meta_parts.append(dur_str)
        src_rows = await db.fetch_all(
            "SELECT connector, external_id FROM track_sources WHERE track_id = :i AND available = TRUE",
            {"i": entity_id},
        )
        img = await images.image_ref(row.get("cover_url"))
        return ShareEntity(
            kind="track",
            entity_id=entity_id,
            title=str(row["title"]),
            subtitle=subtitle,
            meta=" · ".join(meta_parts) or None,
            duration_sec=row.get("duration_sec"),
            image_id=img.id,
            sources=_collect_sources(src_rows),
        )

    if kind == "album":
        row = await db.fetch_one(
            """
            SELECT a.id, a.title, a.year, a.release_date, a.track_count, a.cover_url,
                   ar.name AS artist_name
            FROM albums a
            JOIN artists ar ON ar.id = a.artist_id
            WHERE a.id = :i
            """,
            {"i": entity_id},
        )
        if row is None:
            return None
        year = row.get("year") or (
            row["release_date"].year if row.get("release_date") is not None else None
        )
        meta_parts = []
        if year:
            meta_parts.append(str(year))
        tcount = int(row.get("track_count") or 0)
        if tcount > 0:
            meta_parts.append(f"{tcount} {_ru_plural(tcount, 'трек', 'трека', 'треков')}")
        src_rows = await db.fetch_all(
            "SELECT connector, external_id FROM album_sources WHERE album_id = :i AND available = TRUE",
            {"i": entity_id},
        )
        img = await images.image_ref(row.get("cover_url"))
        return ShareEntity(
            kind="album",
            entity_id=entity_id,
            title=str(row["title"]),
            subtitle=str(row["artist_name"]) if row.get("artist_name") else None,
            meta=" · ".join(meta_parts) or "Альбом",
            duration_sec=None,
            image_id=img.id,
            sources=_collect_sources(src_rows),
        )

    # artist
    row = await db.fetch_one(
        """
        SELECT ar.id, ar.name, ar.cover_url, ar.genre,
               (SELECT COUNT(*) FROM albums a WHERE a.artist_id = ar.id) AS album_count
        FROM artists ar
        WHERE ar.id = :i
        """,
        {"i": entity_id},
    )
    if row is None:
        return None
    meta_parts = []
    if row.get("genre"):
        meta_parts.append(str(row["genre"]).strip())
    acount = int(row.get("album_count") or 0)
    if acount > 0:
        meta_parts.append(f"{acount} {_ru_plural(acount, 'альбом', 'альбома', 'альбомов')}")
    src_rows = await db.fetch_all(
        "SELECT connector, external_id FROM artist_sources WHERE artist_id = :i AND available = TRUE",
        {"i": entity_id},
    )
    img = await images.image_ref(row.get("cover_url"))
    return ShareEntity(
        kind="artist",
        entity_id=entity_id,
        title=str(row["name"]),
        subtitle="Исполнитель",
        meta=" · ".join(meta_parts) or None,
        duration_sec=None,
        image_id=img.id,
        sources=_collect_sources(src_rows),
    )


def _collect_sources(rows: list[dict[str, Any]]) -> dict[str, str]:
    out: dict[str, str] = {}
    for r in rows:
        plat = CONNECTOR_TO_PLATFORM.get(str(r.get("connector") or "").lower())
        eid = str(r.get("external_id") or "").strip()
        if not plat or not eid or _is_synthetic_id(eid):
            continue
        out.setdefault(plat, eid)
    return out


async def _resolve_soundcloud_permalink(kind: ShareKind, external_id: str) -> str | None:
    if not _DIGITS_RE.fullmatch(external_id):
        return None
    resource = {"track": "tracks", "album": "playlists", "artist": "users"}[kind]
    cache_key = f"sc_share:{kind}:{external_id}"
    try:
        from .search_service import cache_get

        cached = await cache_get(cache_key)
        if isinstance(cached, dict) and isinstance(cached.get("url"), str):
            url = cached["url"]
            return url if is_safe_redirect_url("soundcloud", url) else None
    except Exception:  # noqa: BLE001 — optional cache lookup
        pass

    async def _fetch() -> str | None:
        from .. import httpcore
        from ..connectors.soundcloud_direct import _API, _client_id

        client = httpcore.client("soundcloud")
        cid = await _client_id("soundcloud")
        resp = await client.get(
            f"{_API}/{resource}/{external_id}",
            params={"client_id": cid},
            timeout=2.5,
        )
        if resp.status_code in (401, 403):
            cid = await _client_id("soundcloud", force_refresh=True)
            resp = await client.get(
                f"{_API}/{resource}/{external_id}",
                params={"client_id": cid},
                timeout=2.5,
            )
        if resp.status_code != 200:
            return None
        url = str((resp.json() or {}).get("permalink_url") or "").strip()
        if url.startswith("http://"):
            url = "https://" + url[7:]
        return url if is_safe_redirect_url("soundcloud", url) else None

    try:
        url = await asyncio.wait_for(_fetch(), timeout=2.6)
    except Exception as exc:  # noqa: BLE001 — network timeout must fallback cleanly
        log.debug("soundcloud permalink lookup skipped for %s/%s: %s", kind, external_id, exc)
        return None

    if url:
        try:
            from .search_service import cache_put

            await cache_put(cache_key, "soundcloud", {"url": url}, ttl_hours=12)
        except Exception:  # noqa: BLE001
            pass
    return url


def _exact_url_sync(platform: str, kind: ShareKind, eid: str) -> str | None:
    if platform == "yandex" and _DIGITS_RE.fullmatch(eid):
        return f"https://music.yandex.ru/{kind}/{eid}"
    if platform == "apple" and _DIGITS_RE.fullmatch(eid):
        segment = {"track": "song", "album": "album", "artist": "artist"}[kind]
        return f"https://music.apple.com/{segment}/{eid}"
    if platform == "ytmusic":
        if kind == "track" and _YTM_TRACK_RE.fullmatch(eid):
            return f"https://music.youtube.com/watch?v={eid}"
        if kind == "album":
            if _YTM_ALBUM_BROWSE_RE.fullmatch(eid):
                return f"https://music.youtube.com/browse/{eid}"
            if _YTM_ALBUM_LIST_RE.fullmatch(eid):
                return f"https://music.youtube.com/playlist?list={eid}"
        if kind == "artist" and _YTM_ARTIST_RE.fullmatch(eid):
            return f"https://music.youtube.com/channel/{eid}"
    if platform == "vk":
        if kind == "track" and _VK_OWNER_PAIR_RE.fullmatch(eid):
            return f"https://vk.com/audio{eid}"
        if kind == "album" and _VK_OWNER_PAIR_RE.fullmatch(eid):
            return f"https://vk.com/music/album/{eid}"
        if kind == "artist" and _VK_ARTIST_SLUG_RE.fullmatch(eid) and not _DIGITS_RE.fullmatch(eid):
            return f"https://vk.com/artist/{eid}"
    if platform == "deezer" and _DIGITS_RE.fullmatch(eid):
        return f"https://www.deezer.com/{kind}/{eid}"
    if platform == "tidal" and _DIGITS_RE.fullmatch(eid):
        return f"https://tidal.com/browse/{kind}/{eid}"
    if platform == "zvuk" and _DIGITS_RE.fullmatch(eid):
        sub = {"track": "track", "album": "release", "artist": "artist"}[kind]
        return f"https://zvuk.com/{sub}/{eid}"
    return None


def _search_url(platform: str, kind: ShareKind, title: str, subtitle: str | None) -> str:
    query_parts = [p for p in (subtitle if kind != "artist" else None, title) if p]
    q = " — ".join(query_parts) if len(query_parts) > 1 else (query_parts[0] if query_parts else title)
    q_simple = " ".join(query_parts) or title
    eq = quote_plus(q)
    eq_simple = quote_plus(q_simple)

    yandex_types = {"track": "tracks", "album": "albums", "artist": "artists"}[kind]
    sc_scopes = {"track": "sounds", "album": "albums", "artist": "people"}[kind]
    builders: dict[str, str] = {
        "yandex": f"https://music.yandex.ru/search?text={eq}&type={yandex_types}",
        "apple": f"https://music.apple.com/search?term={eq_simple}",
        "ytmusic": f"https://music.youtube.com/search?q={eq}",
        "soundcloud": f"https://soundcloud.com/search/{sc_scopes}?q={eq_simple}",
        "vk": f"https://vk.com/audio?q={eq}",
        "deezer": f"https://www.deezer.com/search/{quote(q_simple, safe='')}",
        "tidal": f"https://listen.tidal.com/search?q={eq_simple}",
        "qobuz": f"https://play.qobuz.com/search?q={eq_simple}",
        "zvuk": f"https://zvuk.com/search?query={eq_simple}",
    }
    return builders[platform]


SoundCloudResolver = Callable[[ShareKind, str], Awaitable[str | None]]


async def resolve_platform_links(
    entity: ShareEntity,
    connected_platforms: set[str] | None = None,
    sc_resolver: SoundCloudResolver | None = None,
) -> list[PlatformLink]:
    """Build exact + fallback search links for all connected services."""
    if connected_platforms is None:
        connected_platforms = await _connected_platforms()

    # Connected services plus any service already linked directly on the entity.
    target_set = set(connected_platforms) | set(entity.sources.keys())
    if not target_set:
        target_set = {"yandex", "apple", "ytmusic", "soundcloud", "vk"}

    sc_lookup = sc_resolver or _resolve_soundcloud_permalink
    ordered = [p for p in PLATFORM_ORDER if p in target_set]

    async def _resolve_one(platform: str) -> PlatformLink | None:
        label = PLATFORM_LABELS.get(platform, platform)
        eid = entity.sources.get(platform)
        exact_url: str | None = None
        if eid:
            if platform == "soundcloud":
                exact_url = await sc_lookup(entity.kind, eid)
            else:
                exact_url = _exact_url_sync(platform, entity.kind, eid)
        if exact_url and is_safe_redirect_url(platform, exact_url):
            return PlatformLink(platform=platform, label=label, url=exact_url, exact=True)
        fallback = _search_url(platform, entity.kind, entity.title, entity.subtitle)
        if not is_safe_redirect_url(platform, fallback):
            return None
        return PlatformLink(platform=platform, label=label, url=fallback, exact=False)

    resolved = [item for item in await asyncio.gather(*(_resolve_one(p) for p in ordered)) if item]
    # Exact catalog links first, calm search fallbacks below (preserving platform order).
    exact_first = [item for item in resolved if item.exact]
    fallbacks = [item for item in resolved if not item.exact]
    return exact_first + fallbacks


def _row_to_record(row: dict[str, Any]) -> ShareRecord:
    raw_links = row.get("links") or []
    if isinstance(raw_links, str):
        raw_links = json.loads(raw_links)
    links: list[PlatformLink] = []
    for item in raw_links:
        if isinstance(item, dict) and item.get("platform") and item.get("url"):
            plat = str(item["platform"])
            url = str(item["url"])
            if is_safe_redirect_url(plat, url):
                links.append(
                    PlatformLink(
                        platform=plat,
                        label=str(item.get("label") or PLATFORM_LABELS.get(plat, plat)),
                        url=url,
                        exact=bool(item.get("exact")),
                    )
                )
    return ShareRecord(
        id=str(row["id"]),
        kind=row["kind"],
        entity_id=int(row["entity_id"]),
        title=str(row["title"]),
        subtitle=row.get("subtitle"),
        meta=row.get("meta"),
        duration_sec=row.get("duration_sec"),
        image_id=str(row.get("image_id") or images.PLACEHOLDER_ID),
        platforms=links,
        clicks_total=int(row.get("clicks_total") or 0),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        expires_at=row["expires_at"],
    )


async def create_or_refresh_share(
    kind: ShareKind,
    entity_id: int,
    user_id: int | None,
    ttl_days: int | None = None,
) -> ShareRecord:
    """Create or slide-extend the single canonical share link for (kind, entity_id)."""
    entity = await _load_entity(kind, entity_id)
    if entity is None:
        raise ProblemError(404, "not_found", f"{kind.capitalize()} not found")

    links = await resolve_platform_links(entity)
    links_json = json.dumps(
        [
            {"platform": item.platform, "label": item.label, "url": item.url, "exact": item.exact}
            for item in links
        ]
    )
    effective_ttl = max(1, int(ttl_days if ttl_days is not None else settings.share_link_ttl_days))
    new_expires_at = datetime.now(UTC) + timedelta(days=effective_ttl)

    async with db.batch():
        existing = await db.fetch_one(
            """
            SELECT * FROM share_links
            WHERE kind = :k AND entity_id = :e
            FOR UPDATE
            """,
            {"k": kind, "e": entity_id},
        )
        now = datetime.now(UTC)
        if existing and existing["expires_at"] > now:
            rows = await db.execute_returning(
                """
                UPDATE share_links
                SET title = :t,
                    subtitle = :s,
                    meta = :m,
                    duration_sec = :d,
                    image_id = :img,
                    links = CAST(:links AS jsonb),
                    updated_at = NOW(),
                    expires_at = :exp
                WHERE id = :id
                RETURNING *
                """,
                {
                    "id": existing["id"],
                    "t": entity.title,
                    "s": entity.subtitle,
                    "m": entity.meta,
                    "d": entity.duration_sec,
                    "img": entity.image_id,
                    "links": links_json,
                    "exp": new_expires_at,
                },
            )
            return _row_to_record(rows[0])

        if existing:
            await db.execute("DELETE FROM share_links WHERE id = :id", {"id": existing["id"]})

        # Rare collision guard across 8-char random IDs.
        share_id = _new_share_id()
        for _ in range(5):
            clash = await db.fetch_one("SELECT 1 FROM share_links WHERE id = :id", {"id": share_id})
            if not clash:
                break
            share_id = _new_share_id()

        rows = await db.execute_returning(
            """
            INSERT INTO share_links
                (id, kind, entity_id, created_by, title, subtitle, meta, duration_sec,
                 image_id, links, clicks_total, created_at, updated_at, expires_at)
            VALUES
                (:id, :k, :e, :u, :t, :s, :m, :d, :img, CAST(:links AS jsonb),
                 0, NOW(), NOW(), :exp)
            RETURNING *
            """,
            {
                "id": share_id,
                "k": kind,
                "e": entity_id,
                "u": user_id,
                "t": entity.title,
                "s": entity.subtitle,
                "m": entity.meta,
                "d": entity.duration_sec,
                "img": entity.image_id,
                "links": links_json,
                "exp": new_expires_at,
            },
        )
        return _row_to_record(rows[0])


async def get_share(share_id: str) -> ShareRecord:
    """Return active ShareRecord or raise ShareNotFoundError / ShareExpiredError."""
    if not _SHARE_ID_RE.fullmatch(share_id or ""):
        raise ShareNotFoundError(share_id)
    row = await db.fetch_one("SELECT * FROM share_links WHERE id = :id", {"id": share_id})
    if row is None:
        raise ShareNotFoundError(share_id)
    rec = _row_to_record(row)
    if rec.is_expired:
        raise ShareExpiredError(rec)
    return rec


async def record_click(share_id: str, platform: str) -> str:
    """Validate platform redirect, bump counters cleanly, and return target URL."""
    rec = await get_share(share_id)
    target = next((p for p in rec.platforms if p.platform == platform), None)
    if target is None or not is_safe_redirect_url(platform, target.url):
        raise ShareNotFoundError(f"{share_id}/{platform}")

    async with db.batch():
        await db.execute(
            "UPDATE share_links SET clicks_total = clicks_total + 1 WHERE id = :id",
            {"id": share_id},
        )
        await db.execute(
            "INSERT INTO share_clicks (share_id, platform) VALUES (:id, :p)",
            {"id": share_id, "p": platform},
        )
    return target.url
