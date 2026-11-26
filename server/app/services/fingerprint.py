"""Fingerprint resolution for manual files (port of kwhale-v2 tagger).

Order: hints > existing tags > Shazam (via VPN) > AcoustID (fpcalc).
Unresolved -> {} (caller quarantines the file).
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

from ..config import settings

log = logging.getLogger("kwhale.fingerprint")

ACOUSTID_URL = "https://api.acoustid.org/v2/lookup"


def _read_existing_tags(filepath: Path) -> dict[str, Any]:
    import mutagen

    mf = mutagen.File(str(filepath), easy=True)
    if mf is None:
        return {}

    def _get(key: str) -> str:
        v = mf.get(key)
        return str(v[0]) if v else ""

    track_number = 0
    if _get("tracknumber"):
        try:
            track_number = int(_get("tracknumber").split("/")[0])
        except ValueError:
            pass
    date = _get("date")
    duration = int(mf.info.length) if getattr(mf, "info", None) and mf.info.length else None
    return {
        "title": _get("title"),
        "artist": _get("artist"),
        "albumartist": _get("albumartist"),
        "album": _get("album"),
        "track_number": track_number,
        "year": int(date[:4]) if date[:4].isdigit() else None,
        "genre": _get("genre"),
        "isrc": _get("isrc") or _get("tsrc"),
        "duration": duration,
    }


async def resolve(filepath: Path, hint: dict[str, Any] | None = None) -> dict[str, Any] | None:
    """Resolve metadata for a file. Returns dict with title/artist or None."""
    meta: dict[str, Any] = {}
    try:
        meta = await asyncio.to_thread(_read_existing_tags, filepath)
    except Exception as e:
        log.debug("tag read %s: %s", filepath, e)

    if hint:
        if hint.get("title"):
            meta["title"] = hint["title"]
        if hint.get("artist"):
            meta["artist"] = hint["artist"]

    if meta.get("title") and meta.get("artist"):
        # existing tags are sufficient
        return meta

    shazam_meta = await _shazam_lookup(filepath)
    if shazam_meta:
        meta = {**shazam_meta, **{k: v for k, v in meta.items() if v}}
        if meta.get("title") and meta.get("artist"):
            meta["_source"] = "shazam"
            return meta

    acoustid_meta = await _acoustid_lookup(filepath)
    if acoustid_meta:
        meta = {**acoustid_meta, **{k: v for k, v in meta.items() if v}}
        if meta.get("title") and meta.get("artist"):
            meta["_source"] = "acoustid"
            return meta

    return meta if (meta.get("title") and meta.get("artist")) else None


async def _shazam_lookup(filepath: Path) -> dict[str, Any] | None:
    try:
        from shazamio import Shazam

        from .. import httpcore

        shazam = Shazam()
        px = httpcore.proxy_for("shazam")
        try:
            out = await shazam.recognize(str(filepath), proxy=px)
        except TypeError:
            out = await shazam.recognize(str(filepath))
        return _parse_shazam(out)
    except Exception as e:
        log.debug("shazam %s: %s", filepath.name, e)
        return None


def _parse_shazam(out: Any) -> dict[str, Any] | None:
    try:
        track = (out or {}).get("track") or {}
        if not track:
            return None
        title = track.get("title")
        subtitle = track.get("subtitle") or ""
        meta: dict[str, Any] = {
            "title": title,
            "artist": subtitle,
            "albumartist": subtitle,
            "cover_url": (track.get("images") or {}).get("coverart") or None,
        }
        # section metadata often carries genre/album
        sections = track.get("sections") or []
        for sec in sections_meta(sections=sections):
            meta.update(sec)
        return meta
    except Exception as e:
        log.debug("shazam parse: %s", e)
        return None


def sections_meta(sections: list[dict]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for sec in sections:
        meta = (sec.get("metadata") or []) if isinstance(sec, dict) else []
        for m in meta:
            if m.get("title") == "Album":
                out.append({"album": m.get("text")})
            elif m.get("title") == "Label":
                out.append({"label": m.get("text")})
            elif m.get("title") == "Released":
                y = str(m.get("text", ""))[:4]
                if y.isdigit():
                    out.append({"year": int(y)})
    return out


async def _acoustid_lookup(filepath: Path) -> dict[str, Any] | None:
    from .. import httpcore

    async def _fpcalc() -> tuple[int, str] | None:
        try:
            proc = await asyncio.create_subprocess_exec(
                "fpcalc",
                "-json",
                str(filepath),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=60.0)
        except FileNotFoundError:
            log.warning("fpcalc not installed")
            return None
        except Exception as e:  # noqa: BLE001
            log.debug("fpcalc %s: %s", filepath.name, e)
            return None
        if proc.returncode != 0:
            return None
        data = json.loads(stdout.decode(errors="ignore"))
        duration = int(data.get("duration") or 0)
        fp = data.get("fingerprint")
        if not fp:
            return None
        return duration, fp

    got = await _fpcalc()
    if not got:
        return None
    duration, fp = got
    try:
        r = await httpcore.client("providers").get(
            ACOUSTID_URL,
            params={
                "client": settings.acoustid_client,
                "duration": duration,
                "fingerprint": fp,
                "meta": "recordings+releases",
            },
            timeout=15.0,
        )
        results = r.json().get("results", [])
    except Exception as e:
        log.warning("acoustid %s: %s", filepath.name, e)
        return None
    if not results:
        return None
    rec = (results[0].get("recordings") or [{}])[0]
    artists = rec.get("artists") or [{}]
    releases = rec.get("releases") or [{}]
    release = releases[0] if releases else {}
    year = (release.get("date") or {}).get("year")
    return {
        "title": rec.get("title", ""),
        "artist": ", ".join(a.get("name", "") for a in artists),
        "albumartist": artists[0].get("name", "") if artists else "",
        "album": release.get("title", ""),
        "year": year,
        "_source": "acoustid",
    }
