"""Content providers: iTunes covers + Genius/Musixmatch lyrics.

Protocols recovered from the RE archive (network/_providers/PROTOCOL.md):
  * iTunes Search  — C2333i.java (covers fallback, rps 0.55 / burst 15)
  * Genius         — C7220i.java (plain lyrics, Android-app headers)
  * Musixmatch     — C3798i.java (lyrics + LRC, usertoken flow, rps 0.5 / 10)

All best-effort: a provider failure never breaks the ingest pipeline —
callers get None and move on. Results cached in search_cache (24h).
"""

from __future__ import annotations

import logging
import re
from typing import Any

from .. import httpcore

log = logging.getLogger("kwhale.providers")

# ── iTunes Search (covers) ───────────────────────────────────────────────────

_OPT_STRIP_RE = re.compile(
    r"\([^)]*\)|\[[^\]]*\]|\b(?:feat|ft)\b\.?|\bprod\.?\s*by\b|\bprod\.?\b", re.IGNORECASE
)


def optimize_search_query(query: str) -> str:
    """RE lib utils optimizeSearchQuery (java:128-133)."""
    q = (query or "").lower()
    q = _OPT_STRIP_RE.sub(" ", q)
    return re.sub(r"\s+", " ", q).strip()


def _fuzzy_ratio(a: str, b: str) -> float:
    """Cheap similarity (0..100) — difflib stands in for fuzzywuzzy's WRatio."""
    import difflib

    return difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio() * 100


async def itunes_cover(artist: str, title: str, album: str | None = None) -> str | None:
    """Best-effort cover URL (300px/1000px) for the track or album."""
    http = httpcore.client("providers")
    term = optimize_search_query(f"{artist} {title}")
    if not term:
        return None
    params: dict[str, Any] = {"term": term, "entity": "song", "limit": 5}
    if re.search(r"[а-яё]", f"{artist}{title}", re.IGNORECASE):
        params["country"] = "RU"
    try:
        await httpcore.throttle("itunes.apple.com")
        r = await http.get("https://itunes.apple.com/search", params=params)
        r.raise_for_status()
        results = r.json().get("results") or []
    except Exception as e:  # noqa: BLE001 — provider is optional
        log.debug("itunes cover lookup failed: %s", e)
        return None
    if not results:
        return None
    want = f"{artist} - {title}".lower()
    scored = sorted(
        results,
        key=lambda x: _fuzzy_ratio(want, f"{x.get('artistName', '')} - {x.get('trackName', '')}"),
        reverse=True,
    )
    best = scored[0]
    if _fuzzy_ratio(want, f"{best.get('artistName', '')} - {best.get('trackName', '')}") < 55:
        return None  # too far off — better no cover than a wrong one
    url = best.get("artworkUrl100") or ""
    return url.replace("100x100", "1000x1000") if url else None


# ── Genius (plain lyrics) ────────────────────────────────────────────────────

_GENIUS_HEADERS = {
    "User-Agent": "Genius/4.2.1 (Android; Android 10; google Pixel 3)",
    "X-Genius-Android-Version": "4.2.1",
}


def _clean_genius_text(text: str) -> str:
    """Section headers get their own paragraph (C7220i.java:534-546)."""
    lines = []
    for raw in text.splitlines():
        line = re.sub(r"\s+", " ", raw).strip()
        if line.startswith("["):
            line = "\n" + line
        lines.append(line)
    out = "\n".join(lines)
    while "\n\n\n" in out:
        out = out.replace("\n\n\n", "\n\n")
    return out.strip()


async def genius_lyrics(artist: str, title: str) -> str | None:
    http = httpcore.client("providers")
    query = f"{artist} - {title}" if artist else title
    api_path = None
    for term in (query, optimize_search_query(query)):
        try:
            r = await http.get(
                "https://api.genius.com/search/song",
                params={"q": term},
                headers=_GENIUS_HEADERS,
            )
            r.raise_for_status()
            sections = (r.json().get("response") or {}).get("sections") or []
            hits = [
                h
                for s in sections
                if s.get("type") == "song"
                for h in (s.get("hits") or [])
            ]
        except Exception as e:  # noqa: BLE001
            log.debug("genius search failed: %s", e)
            return None
        want = query.lower()
        for hit in hits:
            result = hit.get("result") or {}
            pa = ((result.get("primary_artist") or {}).get("name")) or ""
            tag = f"{pa} - {result.get('title')}" if pa else (result.get("title") or "")
            if _fuzzy_ratio(want, tag) >= 70 and result.get("api_path"):
                api_path = result["api_path"]
                break
        if api_path:
            break
    if not api_path:
        return None
    try:
        r = await http.get(
            f"https://api.genius.com{api_path}",
            params={"text_format": "plain"},
            headers=_GENIUS_HEADERS,
        )
        r.raise_for_status()
        song = (r.json().get("response") or {}).get("song") or {}
        plain = ((song.get("lyrics") or {}).get("plain")) or ""
    except Exception as e:  # noqa: BLE001
        log.debug("genius lyrics failed: %s", e)
        return None
    return _clean_genius_text(plain) if plain.strip() else None


# ── Musixmatch (lyrics + LRC) ────────────────────────────────────────────────

_MXM_BASE = "https://apic-appmobile.musixmatch.com/ws/1.1"
_MXM_HEADERS = {
    "User-Agent": "Musixmatch/2025120901 CFNetwork/3860.300.31 Darwin/25.2.0",
    "x-mxm-app-version": "10.1.1",
    "Cookie": "x-mxm-token-guid=",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "application/json",
}
_MXM_APP_ID = "mac-ios-v2.0"
_MXM_BLACKLIST = {115264642, 226291677}  # C3798i.java:70

_LRC_TIME_RE = re.compile(r"^\[(\d+):(\d+)\.(\d+)]")
_LRC_OFFSET_RE = re.compile(r"^\[offset:\s*(-?\d+)\s*]", re.IGNORECASE)
_LRC_WORD_TIMING_RE = re.compile(r"<\d+:\d+\.\d+>\s*")

_mm_token: str | None = None


def parse_lrc(lrc: str) -> list[tuple[int, str]]:
    """LRC text -> [(time_ms, line)] — C3798i.java:548-582."""
    offset_ms = 0
    out: list[tuple[int, str]] = []
    for raw in (lrc or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        m_off = _LRC_OFFSET_RE.match(line)
        if m_off:
            offset_ms = int(m_off.group(1))
            continue
        times: list[int] = []
        while True:
            m = _LRC_TIME_RE.match(line)
            if not m:
                break
            mm, ss, frac = m.group(1), m.group(2), m.group(3)
            ms = int(mm) * 60000 + int(ss) * 1000 + int(frac.ljust(3, "0")[:3])
            times.append(ms)
            line = line[m.end() :]
        if not times:
            continue
        text = _LRC_WORD_TIMING_RE.sub("", line).strip()
        for ms in times:
            out.append((ms + offset_ms, text))
    out.sort(key=lambda x: x[0])
    return out


async def _mxm_token() -> str | None:
    global _mm_token
    if _mm_token:
        return _mm_token
    http = httpcore.client("providers")
    try:
        await httpcore.throttle("musixmatch.com")
        r = await http.get(f"{_MXM_BASE}/token.get", params={"app_id": _MXM_APP_ID}, headers=_MXM_HEADERS)
        r.raise_for_status()
        body = r.json()
        _mm_token = ((body.get("message") or {}).get("body") or {}).get("user_token") or None
    except Exception as e:  # noqa: BLE001
        log.debug("musixmatch token failed: %s", e)
    return _mm_token


async def musixmatch_lyrics(
    artist: str, title: str, duration_sec: int = 0, album: str | None = None
) -> tuple[str, str] | None:
    """-> (plain, lrc) or None. usertoken flow with one 401-retry (§6.3)."""
    global _mm_token
    http = httpcore.client("providers")

    async def _call(token: str) -> dict[str, Any]:
        params: dict[str, Any] = {
            "format": "json",
            "namespace": "lyrics_richsynched",
            "subtitle_format": "lrc",
            "usertoken": token,
            "q_artist": artist,
            "q_track": title,
            "q_duration": str(duration_sec or 0),
            "app_id": _MXM_APP_ID,
        }
        if album:
            params["q_album"] = album
        await httpcore.throttle("musixmatch.com")
        r = await http.get(f"{_MXM_BASE}/macro.subtitles.get", params=params, headers=_MXM_HEADERS)
        r.raise_for_status()
        return r.json()

    token = await _mxm_token()
    if not token:
        return None
    try:
        body = await _call(token)
        status = ((body.get("message") or {}).get("header") or {}).get("status_code")
        if status == 401:  # stale token: refresh once and retry (C3798i.java:322-342)
            _mm_token = None
            token = await _mxm_token()
            if not token:
                return None
            body = await _call(token)
            status = ((body.get("message") or {}).get("header") or {}).get("status_code")
        if status != 200:
            return None
    except Exception as e:  # noqa: BLE001
        log.debug("musixmatch call failed: %s", e)
        return None

    macros = ((body.get("message") or {}).get("body") or {}).get("macro_calls") or {}
    track = (
        ((macros.get("matcher.track.get") or {}).get("message") or {}).get("body") or {}
    ).get("track") or {}
    if track.get("track_id") in _MXM_BLACKLIST:
        return None
    lyrics_body = (
        ((macros.get("track.lyrics.get") or {}).get("message") or {})
        .get("body", {})
        .get("lyrics", {})
        .get("lyrics_body")
        or ""
    )
    lrc_body = (
        (((macros.get("track.subtitles.get") or {}).get("message") or {}).get("body") or {})
        .get("subtitle_list", [{}])[0]
        .get("subtitle", {})
        .get("subtitle_body")
        or ""
    )
    if not lyrics_body.strip() and not lrc_body.strip():
        return None
    return lyrics_body.strip(), lrc_body.strip()


# ── dispatcher ───────────────────────────────────────────────────────────────


async def get_lyrics(
    artist: str,
    title: str,
    duration_sec: int = 0,
    album: str | None = None,
    track_id: int | None = None,
) -> dict[str, str] | None:
    """Lyrics via connectors (Yandex/Deezer LRC first) then providers (Musixmatch -> Genius).

    Returns {"lyrics": plain, "lrc": synced} with whichever parts are found;
    None when nobody has the text. 24h cache. Updates tracks.has_lyrics when found.
    """
    from .. import db
    from ..connectors import get_connector
    from .search_service import cache_get, cache_put

    key = f"lyrics::{artist}::{title}::{album or ''}"
    cached = await cache_get(key)
    if cached is not None:
        if track_id and (cached.get("lyrics") or cached.get("lrc")):
            await db.execute("UPDATE tracks SET has_lyrics = TRUE WHERE id = :i", {"i": track_id})
        return cached

    out: dict[str, str] = {}

    if track_id:
        rows = await db.fetch_all(
            """
            SELECT connector, external_id FROM track_sources
            WHERE track_id = :t AND connector IN ('yandex', 'deezer') AND available
            ORDER BY connector = 'yandex' DESC
            """,
            {"t": track_id},
        )
        for src in rows:
            conn = get_connector(src["connector"])
            if conn is not None and hasattr(conn, "fetch_lyrics"):
                try:
                    got = await conn.fetch_lyrics(src["external_id"])
                    if got:
                        if got.get("lrc") and not out.get("lrc"):
                            out["lrc"] = got["lrc"]
                        if got.get("lyrics") and not out.get("lyrics"):
                            out["lyrics"] = got["lyrics"]
                        if out.get("lrc"):
                            break
                except Exception as e:  # noqa: BLE001
                    log.debug("connector %s lyrics: %s", src["connector"], e)

    if not out.get("lrc"):
        try:
            mxm = await musixmatch_lyrics(artist, title, duration_sec, album)
            if mxm:
                plain_mxm, lrc_mxm = mxm
                if plain_mxm and not out.get("lyrics"):
                    out["lyrics"] = plain_mxm
                if lrc_mxm and not out.get("lrc"):
                    out["lrc"] = lrc_mxm
        except Exception as e:  # noqa: BLE001
            log.debug("musixmatch provider: %s", e)

    if not out.get("lyrics") and not out.get("lrc"):
        try:
            plain = await genius_lyrics(artist, title)
            if plain:
                out["lyrics"] = plain
        except Exception as e:  # noqa: BLE001
            log.debug("genius provider: %s", e)

    if not out:
        return None
    if track_id:
        await db.execute("UPDATE tracks SET has_lyrics = TRUE WHERE id = :i", {"i": track_id})
    await cache_put(key, "providers", out, ttl_hours=24)
    return out
