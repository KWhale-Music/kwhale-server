"""Shared HTTP core for connectors: per-connector clients, per-host throttling.

- client(): cached httpx.AsyncClient with the connector's proxy routing
  (SOURCE_PROXY_<NAME> env; empty = direct) and a desktop-Chrome UA that all
  scraped APIs expect.
- throttle(): token-bucket rate limiter keyed by host. Limits mirror the
  RateLimitInterceptor values recovered from the RE archive (C6760i.java):
  api.vk.ru 1 rps/5 burst, api.deezer.com 5/30, openapi.tidal.com 5/30,
  itunes.apple.com 0.55/15, musixmatch.com 0.5/10.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

log = logging.getLogger("kwhale.httpcore")

CHROME_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)

# host -> (requests per second, burst)
HOST_LIMITS: dict[str, tuple[float, int]] = {
    "api.vk.ru": (1.0, 5),
    "api.deezer.com": (5.0, 30),
    "www.deezer.com": (5.0, 30),
    "media.deezer.com": (5.0, 30),
    "openapi.tidal.com": (5.0, 30),
    "api.tidal.com": (5.0, 30),
    "itunes.apple.com": (0.55, 15),
    "musixmatch.com": (0.5, 10),
    "zvuk.com": (5.0, 30),
}


class _Bucket:
    __slots__ = ("rps", "burst", "tokens", "updated", "lock")

    def __init__(self, rps: float, burst: int) -> None:
        self.rps = rps
        self.burst = burst
        self.tokens = float(burst)
        self.updated = time.monotonic()
        self.lock = asyncio.Lock()


_buckets: dict[str, _Bucket] = {}
_clients: dict[str, Any] = {}

# Canonical connector aliases so yandex_direct -> yandex, etc.
_CANONICAL_CONNECTOR = {
    "yandex_direct": "yandex",
    "soundcloud_direct": "soundcloud",
    "ytm_direct": "ytmusic",
}

# Sources that go direct by default unless explicitly overridden in SOURCE_PROXY_<NAME>
_DIRECT_BY_DEFAULT = {"yandex", "vk", "zvuk", "icm"}

_runtime_proxy: dict[str, Any] | None = None


def _init_runtime_proxy() -> dict[str, Any]:
    global _runtime_proxy
    if _runtime_proxy is None:
        from .config import settings

        _runtime_proxy = {
            "enabled": settings.proxy_enabled,
            "default_proxy": (settings.source_proxy_default or settings.vpn_proxy or "").strip(),
            "overrides": dict(settings.source_proxies),
        }
    return _runtime_proxy


async def load_proxy_settings() -> dict[str, Any]:
    """Load persisted proxy settings from DB (if any) into memory."""
    global _runtime_proxy
    state = _init_runtime_proxy()
    try:
        import json

        from . import db

        row = await db.fetch_one("SELECT enabled, default_proxy, overrides FROM proxy_settings WHERE id = 1")
        if row is not None:
            overrides = row.get("overrides") or {}
            if isinstance(overrides, str):
                overrides = json.loads(overrides)
            state["enabled"] = bool(row["enabled"])
            state["default_proxy"] = (row.get("default_proxy") or "").strip()
            state["overrides"] = {str(k).lower(): str(v).strip() for k, v in overrides.items() if str(v).strip()}
    except Exception as e:  # noqa: BLE001
        log.debug("load_proxy_settings skipped: %s", e)
    return get_proxy_settings()


def get_proxy_settings() -> dict[str, Any]:
    state = _init_runtime_proxy()
    return {
        "enabled": bool(state["enabled"]),
        "defaultProxy": state["default_proxy"],
        "overrides": dict(state["overrides"]),
    }


async def update_proxy_settings(
    enabled: bool,
    default_proxy: str = "",
    overrides: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Persist runtime proxy config and reset all cached HTTP clients."""
    import json

    from . import db

    clean_default = (default_proxy or "").strip()
    clean_overrides = {
        str(k).strip().lower(): str(v).strip()
        for k, v in (overrides or {}).items()
        if str(k).strip() and str(v).strip()
    }
    state = _init_runtime_proxy()
    state["enabled"] = bool(enabled)
    state["default_proxy"] = clean_default
    state["overrides"] = clean_overrides

    await db.execute(
        """
        INSERT INTO proxy_settings (id, enabled, default_proxy, overrides, updated_at)
        VALUES (1, :e, :d, CAST(:o AS jsonb), NOW())
        ON CONFLICT (id) DO UPDATE
        SET enabled = EXCLUDED.enabled,
            default_proxy = EXCLUDED.default_proxy,
            overrides = EXCLUDED.overrides,
            updated_at = NOW()
        """,
        {"e": bool(enabled), "d": clean_default, "o": json.dumps(clean_overrides)},
    )
    await aclose_all()
    return get_proxy_settings()


def _bucket_for(host: str) -> _Bucket:
    b = _buckets.get(host)
    if b is None:
        rps, burst = HOST_LIMITS.get(host, (10.0, 30))
        b = _Bucket(rps, burst)
        _buckets[host] = b
    return b


async def throttle(host: str) -> None:
    """Wait until a request to host fits the rate limit (token bucket)."""
    b = _bucket_for(host)
    async with b.lock:
        while True:
            now = time.monotonic()
            b.tokens = min(b.burst, b.tokens + (now - b.updated) * b.rps)
            b.updated = now
            if b.tokens >= 1.0:
                b.tokens -= 1.0
                return
            await asyncio.sleep((1.0 - b.tokens) / b.rps)


def proxy_for(connector: str) -> str | None:
    """Resolve effective proxy URL for a connector/service, or None for direct connection."""
    from .config import settings

    state = _init_runtime_proxy()
    if not state["enabled"]:
        return None

    canon = _CANONICAL_CONNECTOR.get(connector.lower(), connector.lower())
    overrides: dict[str, str] = state["overrides"]

    for key in (connector.lower(), canon):
        val = overrides.get(key) or settings.source_proxies.get(key)
        if val:
            if val.lower() in ("direct", "none", "off", "false", "0"):
                return None
            return val

    if canon == "shazam" and settings.shazam_proxy:
        if settings.shazam_proxy.lower() not in ("direct", "none", "off"):
            return settings.shazam_proxy

    if canon in _DIRECT_BY_DEFAULT:
        return None

    default_px = state["default_proxy"]
    if default_px and default_px.lower() not in ("direct", "none", "off"):
        return default_px
    return None


def connector_for_url(url: str) -> str:
    """Infer connector/service bucket from a CDN or API URL for proxy routing."""
    u = (url or "").lower()
    if any(h in u for h in ("yandex.net", "yandex.ru", "yastatic.net")):
        return "yandex"
    if any(h in u for h in ("vk.ru", "vk.com", "userapi.com", "vkuseraudio", "vkuserphoto")):
        return "vk"
    if "zvuk.com" in u:
        return "zvuk"
    if any(h in u for h in ("deezer.com", "dzcdn.net")):
        return "deezer"
    if any(h in u for h in ("soundcloud.com", "sndcdn.com")):
        return "soundcloud"
    if any(h in u for h in ("youtube.com", "googlevideo.com", "ytimg.com", "ggpht.com")):
        return "ytmusic"
    if any(h in u for h in ("tidal.com", "wimpmusic.com")):
        return "tidal"
    if "qobuz.com" in u:
        return "qobuz"
    if "byicloud.online" in u:
        return "icm"
    return "providers"


def client(connector: str, *, headers: dict[str, str] | None = None) -> Any:
    """Cached async client for a connector (proxy + Chrome UA + sane timeouts)."""
    import httpx

    px = proxy_for(connector)
    cache_key = f"{connector}::{px or 'direct'}"
    c = _clients.get(cache_key)
    if c is None or c.is_closed:
        merged = {"User-Agent": CHROME_UA, "Accept": "*/*"}
        if headers:
            merged.update(headers)
        c = httpx.AsyncClient(
            timeout=30.0,
            follow_redirects=True,
            proxy=px,
            headers=merged,
        )
        _clients[cache_key] = c
    return c


async def aclose_all() -> None:
    for c in list(_clients.values()):
        try:
            if not c.is_closed:
                await c.aclose()
        except Exception as e:  # noqa: BLE001 — shutdown path
            log.debug("client close: %s", e)
    _clients.clear()
