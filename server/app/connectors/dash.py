"""DASH (MPD) manifest assembly for Tidal streams.

Tidal `playbackinfo` returns a base64 manifest (usually
`application/dash+xml`); The reference implementation concatenates init+media segments into a
single FLAC/AAC stream (container.dash.DashDownloader,
see RE archive network/tidal/PROTOCOL.md §7). Here: parse the MPD into an
ordered segment URL list; the caller downloads and concatenates.

Supports the segment layouts seen in Tidal manifests:
  * SegmentList  -> Initialization@sourceURL + SegmentURL@media[]
  * SegmentTemplate ($Number$, optional startNumber / SegmentTimeline)
  * single BaseURL
Also accepts the `application/vnd.tidal.bts` JSON form (reps[].urls[]).
"""

from __future__ import annotations

import asyncio
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import urljoin

from .base import StreamInfo

_NUM_RE = re.compile(r"\$Number(?::?%0(\d+)d)?\$")
_REP_ID_RE = re.compile(r"\$RepresentationID\$")


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _child(elem: ET.Element, name: str) -> ET.Element | None:
    for c in elem:
        if _local(c.tag) == name:
            return c
    return None


def _children(elem: ET.Element, name: str) -> list[ET.Element]:
    return [c for c in elem if _local(c.tag) == name]


def _expand(template: str, rep_id: str, number: int) -> str:
    out = _REP_ID_RE.sub(rep_id, template)

    def _num(m: re.Match[str]) -> str:
        width = int(m.group(1)) if m.group(1) else 0
        return f"{number:0{width}d}" if width else str(number)

    return _NUM_RE.sub(_num, out)


_ISO_DUR_RE = re.compile(
    r"^PT(?:(\d+(?:\.\d+)?)H)?(?:(\d+(?:\.\d+)?)M)?(?:(\d+(?:\.\d+)?)S)?$"
)


def _iso_seconds(value: str | None) -> float:
    """ISO-8601 duration 'PT1H2M3.5S' -> seconds."""
    if not value:
        return 0.0
    m = _ISO_DUR_RE.match(value.strip())
    if not m:
        return 0.0
    h, mnt, s = (float(g) if g else 0.0 for g in m.groups())
    return h * 3600 + mnt * 60 + s


def parse_manifest(manifest_text: str, base_url: str = "") -> list[str]:
    """Manifest (MPD XML or BTS JSON) -> ordered segment URLs (init first)."""
    text = manifest_text.strip()
    if text.startswith("{"):
        return _parse_bts(text, base_url)
    return _parse_mpd(text, base_url)


def decode_data_uri(uri: str) -> tuple[str, str]:
    """`data:<mime>;base64,<payload>` -> (mime, decoded text)."""
    import base64

    if not uri.startswith("data:"):
        raise ValueError("not a data URI")
    head, _, payload = uri.partition(",")
    mime = head[5:].split(";")[0]
    if ";base64" in head:
        return mime, base64.b64decode(payload).decode("utf-8", "replace")
    from urllib.parse import unquote

    return mime, unquote(payload)


async def _fetch_segment(client, url: str, sem: asyncio.Semaphore) -> bytes:
    async with sem:
        async with client.stream("GET", url) as r:
            r.raise_for_status()
            chunks = [chunk async for chunk in r.aiter_bytes(64 * 1024)]
            return b"".join(chunks)


async def fetch_concat(client, urls: list[str], dest, *, concurrency: int = 4) -> None:
    """Download segments with bounded concurrency and concatenate in strict order into dest (Path)."""
    if not urls:
        raise ValueError("no segments to fetch")
    sem = asyncio.Semaphore(max(1, concurrency))
    window = max(1, concurrency) * 2
    with open(dest, "wb") as f:
        for offset in range(0, len(urls), window):
            batch = urls[offset : offset + window]
            tasks = [asyncio.create_task(_fetch_segment(client, u, sem)) for u in batch]
            try:
                for task in tasks:
                    f.write(await task)
            except Exception:
                for task in tasks:
                    if not task.done():
                        task.cancel()
                raise


async def assemble_concat(connector_name: str, info: StreamInfo, dest: Path) -> None:
    """Assemble a segment stream into a single file at dest.

    Accepts either pre-parsed info.segment_urls or a manifest in info.url
    (data-uri / BTS / MPD). Shared by the concat-style connectors (tidal
    'dash_concat', soundcloud 'hls_concat') — each calls this from its
    Connector.assemble_stream() implementation.
    """
    from .. import httpcore

    if info.segment_urls:
        urls = info.segment_urls
    elif info.url:
        _mime, text = decode_data_uri(info.url)
        urls = parse_manifest(text)
    else:
        urls = []
    if not urls:
        raise ValueError("empty segment list")
    http = httpcore.client(connector_name)
    await fetch_concat(http, urls, dest)


def _parse_bts(text: str, base_url: str) -> list[str]:
    root = json.loads(text)
    reps = root.get("reps") or []
    if not reps:
        return []
    # audio-only: pick the representation with the most urls (usually just one)
    best = max(reps, key=lambda r: len(r.get("urls") or []))
    return [urljoin(base_url, u) for u in (best.get("urls") or [])]


def _parse_mpd(text: str, base_url: str) -> list[str]:
    root = ET.fromstring(text)  # noqa: S314 — manifest from Tidal, not user input
    mpd_base = base_url
    base_elem = _child(root, "BaseURL")
    if base_elem is not None and base_elem.text:
        mpd_base = urljoin(base_url, base_elem.text.strip())

    # audio adaptation set -> representation (highest bandwidth wins)
    reps: list[ET.Element] = []
    for ads in root.iter():
        if _local(ads.tag) != "AdaptationSet":
            continue
        mime = (ads.get("mimeType") or "").lower()
        reps_here = [r for a in _children(ads, "Representation") for r in [a]]
        if reps_here and (not mime or mime.startswith("audio") or "audio" in mime or mime == ""):
            reps.extend(reps_here)
    if not reps:
        for r in root.iter():
            if _local(r.tag) == "Representation":
                reps.append(r)
    if not reps:
        return []

    def _bandwidth(r: ET.Element) -> int:
        try:
            return int(r.get("bandwidth") or 0)
        except ValueError:
            return 0

    rep = max(reps, key=_bandwidth)
    rep_id = rep.get("id") or "1"
    rep_base = mpd_base
    rep_base_elem = _child(rep, "BaseURL")
    if rep_base_elem is not None and rep_base_elem.text:
        rep_base = urljoin(rep_base, rep_base_elem.text.strip())

    urls: list[str] = []

    seg_list = _child(rep, "SegmentList")
    if seg_list is not None:
        init = _child(seg_list, "Initialization")
        if init is not None and init.get("sourceURL"):
            urls.append(urljoin(rep_base, init.get("sourceURL", "")))
        for seg in _children(seg_list, "SegmentURL"):
            media = seg.get("media")
            if media:
                urls.append(urljoin(rep_base, media))
        return urls

    seg_tpl = _child(rep, "SegmentTemplate")
    if seg_tpl is not None:
        init_tpl = (seg_tpl.get("initialization") or "").strip()
        media_tpl = (seg_tpl.get("media") or "").strip()
        start = int(seg_tpl.get("startNumber") or 1)
        if init_tpl:
            urls.append(urljoin(rep_base, _expand(init_tpl, rep_id, start)))
        if media_tpl:
            timeline = _child(seg_tpl, "SegmentTimeline")
            number = start
            if timeline is not None:
                for s in _children(timeline, "S"):
                    count = int(s.get("r") or 0) + 1
                    for _ in range(count):
                        urls.append(urljoin(rep_base, _expand(media_tpl, rep_id, number)))
                        number += 1
            else:
                # no timeline: derive the segment count from period/segment durations
                seg_dur = float(seg_tpl.get("segmentDuration") or rep.get("duration") or 0)
                timescale = float(seg_tpl.get("timescale") or rep.get("timescale") or 1) or 1.0
                period = _child(root, "Period")
                total = _iso_seconds(period.get("duration")) if period is not None else 0.0
                count = int(total / (seg_dur / timescale)) if seg_dur and total else 1
                for i in range(max(1, count)):
                    urls.append(urljoin(rep_base, _expand(media_tpl, rep_id, start + i)))
        return urls

    # single BaseURL on the representation or period
    if rep_base_elem is not None and rep_base_elem.text and not urls:
        urls.append(urljoin(mpd_base, rep_base_elem.text.strip()))
    if not urls and rep_base != mpd_base:
        urls.append(rep_base)
    return urls
