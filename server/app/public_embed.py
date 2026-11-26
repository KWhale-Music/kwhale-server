"""Unauthenticated Yandex-style iframe embed cards with KWhale M3 Monet visuals."""

from __future__ import annotations

import html
import json
from typing import Any

from fastapi import APIRouter, Query, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse

from . import auth
from .errors import ProblemError
from .public_share import FONT_FACE_CSS
from .services import embed_service, share_card, share_service

router = APIRouter(prefix="/share/embed", tags=["embed-public"])

_KIND_BADGES: dict[str, str] = {
    "track": "Трек",
    "album": "Альбом",
    "artist": "Исполнитель",
}

_PLAY_SVG = (
    '<svg viewBox="0 0 24 24" aria-hidden="true">'
    '<path d="M8 5.14v13.72a1 1 0 0 0 1.52.85l10.6-6.86a1 1 0 0 0 0-1.7L9.52 4.29A1 1 0 0 0 8 5.14z" fill="currentColor"/>'
    "</svg>"
)
_PAUSE_SVG = (
    '<svg viewBox="0 0 24 24" aria-hidden="true">'
    '<path d="M7 5h3.5v14H7V5zm6.5 0H17v14h-3.5V5z" fill="currentColor"/>'
    "</svg>"
)
_PREV_SVG = (
    '<svg viewBox="0 0 24 24" aria-hidden="true">'
    '<path d="M6 6h2v12H6V6zm3.5 6l8.5 6V6l-8.5 6z" fill="currentColor"/>'
    "</svg>"
)
_NEXT_SVG = (
    '<svg viewBox="0 0 24 24" aria-hidden="true">'
    '<path d="M6 18l8.5-6L6 6v12zM16 6v12h2V6h-2z" fill="currentColor"/>'
    "</svg>"
)
_VOL_SVG = (
    '<svg viewBox="0 0 24 24" aria-hidden="true">'
    '<path d="M3 9v6h4l5 5V4L7 9H3zm13.5 3A4.5 4.5 0 0 0 14 7.97v8.05c1.48-.73 2.5-2.25 2.5-4.02z'
    'M14 3.23v2.06c2.89.86 5 3.54 5 6.71s-2.11 5.85-5 6.71v2.06c4.01-.91 7-4.49 7-8.77s-2.99-7.86-7-8.77z" fill="currentColor"/>'
    "</svg>"
)
_MUTE_SVG = (
    '<svg viewBox="0 0 24 24" aria-hidden="true">'
    '<path d="M16.5 12A4.5 4.5 0 0 0 14 7.97v2.21l2.45 2.45c.03-.2.05-.41.05-.63zm2.5 0c0 .94-.2 1.82-.54 2.64'
    'l1.51 1.51C20.63 14.91 21 13.5 21 12c0-4.28-2.99-7.86-7-8.77v2.06c2.89.86 5 3.54 5 6.71zM4.27 3L3 4.27 7.73 9H3v6h4'
    'l5 5v-6.73l4.25 4.25c-.67.52-1.42.93-2.25 1.18v2.06c1.38-.31 2.63-.95 3.69-1.81L19.73 21 21 19.73l-9-9L4.27 3zM12 4L9.91 6.09 12 8.18V4z" fill="currentColor"/>'
    "</svg>"
)


def _theme_css(tokens: dict[str, dict[str, Any]], theme: str) -> str:
    dark_vars = "; ".join(f"{k}: {v}" for k, v in tokens["dark"]["css"].items())
    light_vars = "; ".join(f"{k}: {v}" for k, v in tokens["light"]["css"].items())
    if theme == "dark":
        return f":root {{ color-scheme: dark; {dark_vars}; }}"
    if theme == "light":
        return f":root {{ color-scheme: light; {light_vars}; }}"
    return (
        f":root {{ color-scheme: dark light; {dark_vars}; }}\n"
        f"@media (prefers-color-scheme: light) {{ :root {{ {light_vars}; }} }}"
    )


def _embed_styles(token_css: str) -> str:
    return f"""
<style>
{FONT_FACE_CSS}
{token_css}
*, *::before, *::after {{ box-sizing: border-box; }}
html, body {{
  margin: 0;
  padding: 0;
  width: 100%;
  height: 100%;
  overflow: hidden;
  background: transparent;
  color: var(--md-sys-color-on-surface);
  font-family: 'Roboto', system-ui, -apple-system, 'Segoe UI', sans-serif;
  -webkit-font-smoothing: antialiased;
}}
.widget {{
  width: 100%;
  height: 100%;
  display: flex;
  flex-direction: column;
  background-color: var(--md-sys-color-surface-container-low);
  background-image: radial-gradient(
    ellipse 120% 220px at 0% 0%,
    color-mix(in srgb, var(--md-sys-color-primary-container) 44%, transparent) 0%,
    transparent 74%
  );
  border: 1px solid color-mix(in srgb, var(--md-sys-color-outline-variant) 76%, transparent);
  border-radius: 20px;
  overflow: hidden;
  position: relative;
}}
.head {{
  display: flex;
  align-items: center;
  gap: 14px;
  padding: 14px 16px 10px;
  min-width: 0;
  flex-shrink: 0;
}}
.cover-wrap {{
  position: relative;
  width: 68px;
  height: 68px;
  flex-shrink: 0;
  border-radius: 14px;
  background: var(--md-sys-color-surface-container-high);
  border: 1px solid color-mix(in srgb, var(--md-sys-color-outline-variant) 68%, transparent);
  overflow: hidden;
  cursor: pointer;
  padding: 0;
}}
.widget.is-artist .cover-wrap {{
  border-radius: 9999px;
}}
.cover-img {{
  width: 100%;
  height: 100%;
  object-fit: cover;
  display: block;
}}
.cover-scrim {{
  position: absolute;
  inset: 0;
  background: rgba(0, 0, 0, 0.36);
  color: #fff;
  display: flex;
  align-items: center;
  justify-content: center;
  opacity: 0;
  transition: opacity 150ms ease;
}}
.cover-wrap:hover .cover-scrim,
.cover-wrap:focus-visible .cover-scrim {{
  opacity: 1;
}}
.cover-scrim svg {{
  width: 28px;
  height: 28px;
}}
.head-meta {{
  min-width: 0;
  flex: 1;
  display: flex;
  flex-direction: column;
  gap: 3px;
}}
.head-top {{
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
}}
.kind-chip {{
  font-size: 11px;
  line-height: 16px;
  font-weight: 600;
  letter-spacing: 0.02em;
  text-transform: uppercase;
  color: var(--md-sys-color-primary);
}}
.brand-pill {{
  display: inline-flex;
  align-items: center;
  gap: 5px;
  padding: 4px 10px;
  border-radius: 9999px;
  background: color-mix(in srgb, var(--md-sys-color-surface-container-highest) 78%, transparent);
  border: 1px solid color-mix(in srgb, var(--md-sys-color-outline-variant) 64%, transparent);
  color: var(--md-sys-color-on-surface-variant);
  text-decoration: none;
  font-size: 11px;
  font-weight: 500;
  white-space: nowrap;
  flex-shrink: 0;
  transition: background-color 140ms ease, color 140ms ease;
}}
.brand-pill:hover {{
  background: var(--md-sys-color-primary-container);
  color: var(--md-sys-color-on-primary-container);
}}
.brand-mark {{
  width: 14px;
  height: 14px;
  background-color: currentColor;
  -webkit-mask: url('/share/static/kwhale-mark.png') center / contain no-repeat;
  mask: url('/share/static/kwhale-mark.png') center / contain no-repeat;
}}
.title {{
  margin: 0;
  font-size: 17px;
  line-height: 22px;
  font-weight: 700;
  letter-spacing: -0.015em;
  color: var(--md-sys-color-on-surface);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}}
.subtitle {{
  margin: 0;
  font-size: 13px;
  line-height: 18px;
  font-weight: 500;
  color: var(--md-sys-color-on-surface-variant);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}}
.controls {{
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 6px 16px 14px;
  flex-shrink: 0;
}}
.widget.is-full .controls {{
  padding: 4px 16px 12px;
  border-bottom: 1px solid color-mix(in srgb, var(--md-sys-color-outline-variant) 55%, transparent);
}}
.play-btn {{
  width: 42px;
  height: 42px;
  border-radius: 9999px;
  border: 0;
  padding: 0;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  background: var(--md-sys-color-primary);
  color: var(--md-sys-color-on-primary);
  cursor: pointer;
  flex-shrink: 0;
  transition: transform 110ms ease, filter 140ms ease;
}}
.play-btn:hover {{ filter: brightness(1.06); }}
.play-btn:active {{ transform: scale(0.94); }}
.play-btn svg {{ width: 22px; height: 22px; display: block; }}
.icon-btn {{
  width: 32px;
  height: 32px;
  border-radius: 9999px;
  border: 0;
  padding: 0;
  background: transparent;
  color: var(--md-sys-color-on-surface-variant);
  display: inline-flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  flex-shrink: 0;
  transition: background-color 140ms ease, color 140ms ease;
}}
.icon-btn:hover {{
  background: color-mix(in srgb, var(--md-sys-color-on-surface) 8%, transparent);
  color: var(--md-sys-color-on-surface);
}}
.icon-btn:disabled {{ opacity: 0.35; cursor: default; background: transparent; }}
.icon-btn svg {{ width: 19px; height: 19px; display: block; }}
.progress-col {{
  flex: 1;
  min-width: 0;
  display: flex;
  flex-direction: column;
  gap: 4px;
}}
.now-playing {{
  font-size: 12px;
  line-height: 16px;
  font-weight: 500;
  color: var(--md-sys-color-on-surface);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  display: none;
}}
.widget.has-multi .now-playing {{ display: block; }}
.seek-row {{
  display: flex;
  align-items: center;
  gap: 8px;
}}
input[type=range].seek {{
  -webkit-appearance: none;
  appearance: none;
  flex: 1;
  min-width: 40px;
  height: 20px;
  margin: 0;
  background: transparent;
  cursor: pointer;
  --pct: 0%;
}}
input[type=range].seek:focus {{ outline: none; }}
input[type=range].seek::-webkit-slider-runnable-track {{
  height: 5px;
  border-radius: 9999px;
  background: linear-gradient(
    to right,
    var(--md-sys-color-primary) 0%,
    var(--md-sys-color-primary) var(--pct),
    color-mix(in srgb, var(--md-sys-color-outline-variant) 78%, transparent) var(--pct),
    color-mix(in srgb, var(--md-sys-color-outline-variant) 78%, transparent) 100%
  );
}}
input[type=range].seek::-webkit-slider-thumb {{
  -webkit-appearance: none;
  appearance: none;
  width: 13px;
  height: 13px;
  margin-top: -4px;
  border-radius: 9999px;
  background: var(--md-sys-color-primary);
  box-shadow: 0 1px 4px rgba(0, 0, 0, 0.35);
}}
input[type=range].seek::-moz-range-track {{
  height: 5px;
  border-radius: 9999px;
  background: color-mix(in srgb, var(--md-sys-color-outline-variant) 78%, transparent);
}}
input[type=range].seek::-moz-range-progress {{
  height: 5px;
  border-radius: 9999px;
  background: var(--md-sys-color-primary);
}}
input[type=range].seek::-moz-range-thumb {{
  width: 13px;
  height: 13px;
  border: 0;
  border-radius: 9999px;
  background: var(--md-sys-color-primary);
}}
.time-readout {{
  font-size: 11px;
  line-height: 14px;
  font-variant-numeric: tabular-nums;
  color: var(--md-sys-color-on-surface-variant);
  white-space: nowrap;
  flex-shrink: 0;
}}
.tracklist {{
  flex: 1;
  min-height: 0;
  overflow-y: auto;
  padding: 6px 8px 10px;
  scrollbar-width: thin;
  scrollbar-color: var(--md-sys-color-outline-variant) transparent;
}}
.trow {{
  width: 100%;
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 7px 10px;
  border: 0;
  background: transparent;
  color: var(--md-sys-color-on-surface);
  border-radius: 12px;
  cursor: pointer;
  text-align: left;
  font: inherit;
  transition: background-color 120ms ease;
}}
.trow:hover {{
  background: color-mix(in srgb, var(--md-sys-color-on-surface) 6%, transparent);
}}
.trow.active {{
  background: color-mix(in srgb, var(--md-sys-color-primary-container) 62%, transparent);
}}
.tnum {{
  width: 22px;
  text-align: center;
  font-size: 12px;
  font-variant-numeric: tabular-nums;
  color: var(--md-sys-color-on-surface-variant);
  flex-shrink: 0;
}}
.trow.active .tnum {{
  color: var(--md-sys-color-primary);
  font-weight: 700;
}}
.tthumb {{
  width: 34px;
  height: 34px;
  border-radius: 8px;
  object-fit: cover;
  flex-shrink: 0;
  background: var(--md-sys-color-surface-container-high);
}}
.tinfo {{
  min-width: 0;
  flex: 1;
}}
.ttitle {{
  font-size: 13px;
  line-height: 18px;
  font-weight: 500;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}}
.trow.active .ttitle {{
  color: var(--md-sys-color-on-primary-container);
  font-weight: 600;
}}
.tartist {{
  font-size: 11px;
  line-height: 15px;
  color: var(--md-sys-color-on-surface-variant);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}}
.tdur {{
  font-size: 12px;
  font-variant-numeric: tabular-nums;
  color: var(--md-sys-color-on-surface-variant);
  flex-shrink: 0;
}}
.eq {{
  display: inline-flex;
  align-items: flex-end;
  gap: 2px;
  height: 12px;
}}
.eq span {{
  width: 2.5px;
  border-radius: 2px;
  background: var(--md-sys-color-primary);
  animation: eqPulse 0.8s ease-in-out infinite alternate;
}}
.eq span:nth-child(2) {{ animation-delay: 0.2s; }}
.eq span:nth-child(3) {{ animation-delay: 0.4s; }}
@keyframes eqPulse {{
  from {{ height: 3px; }}
  to {{ height: 12px; }}
}}
@media (prefers-reduced-motion: reduce) {{
  .eq span {{ animation: none; height: 8px; }}
}}
.status-msg {{
  position: absolute;
  left: 16px;
  right: 16px;
  bottom: 8px;
  padding: 6px 10px;
  border-radius: 10px;
  background: color-mix(in srgb, var(--md-sys-color-error-container) 92%, transparent);
  color: var(--md-sys-color-on-error-container);
  font-size: 12px;
  display: none;
}}
.status-msg.show {{ display: block; }}
.dead {{
  width: 100%;
  height: 100%;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  text-align: center;
  padding: 20px;
  gap: 6px;
}}
@media (max-width: 380px) {{
  .vol-btn {{ display: none; }}
  .cover-wrap {{ width: 56px; height: 56px; }}
  .head {{ padding: 12px 12px 8px; gap: 10px; }}
  .controls {{ padding: 4px 12px 12px; gap: 8px; }}
}}
</style>
"""


def _headers(allowed_origins: list[str] | None = None) -> dict[str, str]:
    csp = embed_service.frame_ancestors_csp(allowed_origins or [])
    return {
        "Cache-Control": "no-store",
        "Content-Security-Policy": csp,
        "Vary": "Accept-Encoding",
    }


def _render_embed_status(title: str, subtitle: str, status_code: int) -> HTMLResponse:
    seed = share_card.PaletteSeed(hue_deg=214.0, saturation=0.45)
    tokens = share_card.build_monet_tokens(seed)
    css = _embed_styles(_theme_css(tokens, "auto"))
    doc = f"""<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)} — KWhale</title>
{css}
</head>
<body>
<div class="widget">
  <div class="dead">
    <span class="brand-mark" style="width:24px;height:24px;color:var(--md-sys-color-primary)"></span>
    <p class="title" style="white-space:normal">{html.escape(title)}</p>
    <p class="subtitle" style="white-space:normal">{html.escape(subtitle)}</p>
  </div>
</div>
</body>
</html>"""
    return HTMLResponse(content=doc, status_code=status_code, headers=_headers([]))


@router.get("/{widget_id}", response_class=HTMLResponse)
async def render_embed_widget(
    widget_id: str,
    theme: str | None = Query(None),
    layout: str | None = Query(None),
) -> HTMLResponse:
    try:
        rec = await embed_service.get_public_widget(widget_id)
    except embed_service.EmbedInactiveError:
        return _render_embed_status("Интеграция отключена", "Виджет отозван администратором KWhale.", 410)
    except embed_service.EmbedNotFoundError:
        return _render_embed_status("Карточка не найдена", "Проверьте ссылку embed-интеграции.", 404)

    eff_theme = theme if theme in ("auto", "dark", "light") else rec.theme
    req_layout = layout if layout in ("auto", "compact", "full") else rec.layout
    is_full = req_layout == "full" or (req_layout == "auto" and rec.kind in ("album", "artist") and len(rec.tracks) > 1)

    tokens = await share_card.get_monet_tokens(rec.image_id, rec.subtitle or rec.title)
    css = _embed_styles(_theme_css(tokens, eff_theme))

    track_payload: list[dict[str, Any]] = []
    list_rows: list[str] = []
    for idx, t in enumerate(rec.tracks):
        stream_path = f"/v1/tracks/{t.id}/stream"
        exp, sig = auth.sign_media_path(stream_path, ttl_sec=3600)
        cover_src = f"/v1/images/{t.image_id}?size=xs"
        track_payload.append(
            {
                "id": str(t.id),
                "title": t.title,
                "artists": t.artists or rec.subtitle or "",
                "durationSec": t.duration_sec or 0,
                "coverUrl": f"/v1/images/{t.image_id}?size=md",
                "streamUrl": f"{stream_path}?exp={exp}&sig={sig}",
                "expiresAt": exp,
            }
        )
        if is_full and len(rec.tracks) > 1:
            dur = share_service.format_duration(t.duration_sec) or "—:—"
            list_rows.append(
                f'<button type="button" class="trow" data-idx="{idx}">'
                f'<span class="tnum">{idx + 1}</span>'
                f'<img class="tthumb" src="{html.escape(cover_src, quote=True)}" alt="" loading="lazy">'
                f'<span class="tinfo">'
                f'<div class="ttitle">{html.escape(t.title)}</div>'
                f'<div class="tartist">{html.escape(t.artists or rec.subtitle or "")}</div>'
                f"</span>"
                f'<span class="tdur">{html.escape(dur)}</span>'
                f"</button>"
            )

    hero_cover = f"/v1/images/{rec.image_id}?size=md"
    open_href = f"/share/embed/{html.escape(rec.id, quote=True)}/open"
    kind_badge = _KIND_BADGES.get(rec.kind, "Аудио")
    subtitle_text = rec.subtitle or (f"{len(rec.tracks)} треков" if len(rec.tracks) > 1 else "KWhale")

    classes = ["widget"]
    if rec.kind == "artist":
        classes.append("is-artist")
    if is_full:
        classes.append("is-full")
    if len(rec.tracks) > 1:
        classes.append("has-multi")

    nav_btns = (
        f'<button type="button" id="prev-btn" class="icon-btn" aria-label="Предыдущий трек">{_PREV_SVG}</button>'
        if len(rec.tracks) > 1
        else ""
    )
    next_btn = (
        f'<button type="button" id="next-btn" class="icon-btn" aria-label="Следующий трек">{_NEXT_SVG}</button>'
        if len(rec.tracks) > 1
        else ""
    )
    tracklist_html = f'<div id="tracklist" class="tracklist" role="list">{"".join(list_rows)}</div>' if list_rows else ""

    init_dur = share_service.format_duration(rec.tracks[0].duration_sec) if rec.tracks else None
    init_time = f"0:00 / {init_dur or '0:00'}"

    doc = f"""<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(rec.title)} — KWhale</title>
{css}
</head>
<body>
<div id="widget" class="{' '.join(classes)}">
  <header class="head">
    <button type="button" id="cover-btn" class="cover-wrap" aria-label="Воспроизвести или поставить на паузу">
      <img id="cover-img" class="cover-img" src="{html.escape(hero_cover, quote=True)}" alt="{html.escape(rec.title, quote=True)}">
      <span id="cover-scrim" class="cover-scrim">{_PLAY_SVG}</span>
    </button>
    <div class="head-meta">
      <div class="head-top">
        <span class="kind-chip">{html.escape(kind_badge)}</span>
        <a class="brand-pill" href="{open_href}" target="_blank" rel="noopener noreferrer" title="Открыть в KWhale">
          <span class="brand-mark" aria-hidden="true"></span>
          <span>KWhale</span>
        </a>
      </div>
      <h1 class="title" title="{html.escape(rec.title, quote=True)}">{html.escape(rec.title)}</h1>
      <p class="subtitle" title="{html.escape(subtitle_text, quote=True)}">{html.escape(subtitle_text)}</p>
    </div>
  </header>

  <div class="controls">
    {nav_btns}
    <button type="button" id="play-btn" class="play-btn" aria-label="Воспроизвести">{_PLAY_SVG}</button>
    {next_btn}
    <div class="progress-col">
      <div id="now-title" class="now-playing">{html.escape(rec.tracks[0].title if rec.tracks else rec.title)}</div>
      <div class="seek-row">
        <input id="seek" class="seek" type="range" min="0" max="1000" value="0" aria-label="Перемотка">
        <span id="time-readout" class="time-readout">{html.escape(init_time)}</span>
      </div>
    </div>
    <button type="button" id="vol-btn" class="icon-btn vol-btn" aria-label="Звук">{_VOL_SVG}</button>
  </div>

  {tracklist_html}
  <div id="err-box" class="status-msg" role="alert"></div>
  <audio id="audio" preload="none" crossorigin="anonymous"></audio>
</div>

<script>
(() => {{
  const WIDGET_ID = {json.dumps(rec.id)};
  const TRACKS = {json.dumps(track_payload, ensure_ascii=False)};
  const PLAY_SVG = {json.dumps(_PLAY_SVG)};
  const PAUSE_SVG = {json.dumps(_PAUSE_SVG)};
  const VOL_SVG = {json.dumps(_VOL_SVG)};
  const MUTE_SVG = {json.dumps(_MUTE_SVG)};

  const audio = document.getElementById("audio");
  const playBtn = document.getElementById("play-btn");
  const coverBtn = document.getElementById("cover-btn");
  const coverScrim = document.getElementById("cover-scrim");
  const coverImg = document.getElementById("cover-img");
  const prevBtn = document.getElementById("prev-btn");
  const nextBtn = document.getElementById("next-btn");
  const seek = document.getElementById("seek");
  const timeEl = document.getElementById("time-readout");
  const nowEl = document.getElementById("now-title");
  const volBtn = document.getElementById("vol-btn");
  const errBox = document.getElementById("err-box");
  const rows = [...document.querySelectorAll(".trow")];

  let idx = 0;
  let seeking = false;
  let retriedRefresh = false;

  function fmt(sec) {{
    if (!Number.isFinite(sec) || sec < 0) return "0:00";
    const s = Math.floor(sec);
    const h = Math.floor(s / 3600);
    const m = Math.floor((s % 3600) / 60);
    const r = s % 60;
    return h ? `${{h}}:${{String(m).padStart(2, "0")}}:${{String(r).padStart(2, "0")}}` : `${{m}}:${{String(r).padStart(2, "0")}}`;
  }}

  function showErr(msg) {{
    if (!msg) {{ errBox.classList.remove("show"); errBox.textContent = ""; return; }}
    errBox.textContent = msg;
    errBox.classList.add("show");
  }}

  function renderActiveRow() {{
    const playing = !audio.paused && !audio.ended;
    playBtn.innerHTML = playing ? PAUSE_SVG : PLAY_SVG;
    coverScrim.innerHTML = playing ? PAUSE_SVG : PLAY_SVG;
    rows.forEach((row, i) => {{
      const isCur = i === idx;
      row.classList.toggle("active", isCur);
      const num = row.querySelector(".tnum");
      if (num) {{
        num.innerHTML = (isCur && playing)
          ? '<span class="eq"><span></span><span></span><span></span></span>'
          : String(i + 1);
      }}
    }});
  }}

  async function ensureFreshUrl(track) {{
    const nowSec = Math.floor(Date.now() / 1000);
    if (track.streamUrl && (!track.expiresAt || track.expiresAt - nowSec > 90)) {{
      return track.streamUrl;
    }}
    const resp = await fetch(`/share/embed/${{encodeURIComponent(WIDGET_ID)}}/stream/${{encodeURIComponent(track.id)}}`, {{ cache: "no-store" }});
    if (!resp.ok) throw new Error(resp.status === 410 ? "Интеграция отключена" : `HTTP ${{resp.status}}`);
    const data = await resp.json();
    track.streamUrl = data.streamUrl;
    track.expiresAt = data.expiresAt;
    return track.streamUrl;
  }}

  function selectTrack(newIdx, autoplay) {{
    if (!TRACKS.length) return;
    idx = (newIdx + TRACKS.length) % TRACKS.length;
    retriedRefresh = false;
    const t = TRACKS[idx];
    nowEl.textContent = t.artists ? `${{t.title}} — ${{t.artists}}` : t.title;
    if (t.coverUrl) coverImg.src = t.coverUrl;
    seek.value = "0";
    seek.style.setProperty("--pct", "0%");
    timeEl.textContent = `0:00 / ${{fmt(t.durationSec)}}`;
    showErr("");
    if (autoplay) {{
      playCurrent();
    }} else {{
      audio.removeAttribute("src");
      audio.load();
      renderActiveRow();
    }}
  }}

  async function playCurrent() {{
    const t = TRACKS[idx];
    if (!t) return;
    showErr("");
    try {{
      const url = await ensureFreshUrl(t);
      if (!audio.getAttribute("src") || audio.getAttribute("src") !== url) {{
        audio.src = url;
      }}
      await audio.play();
    }} catch (err) {{
      showErr((err && err.message) || "Не удалось воспроизвести трек");
      renderActiveRow();
    }}
  }}

  function togglePlay() {{
    if (!TRACKS.length) return;
    if (audio.paused || audio.ended) {{
      playCurrent();
    }} else {{
      audio.pause();
    }}
  }}

  playBtn.addEventListener("click", togglePlay);
  coverBtn.addEventListener("click", togglePlay);
  if (prevBtn) prevBtn.addEventListener("click", () => selectTrack(idx - 1, true));
  if (nextBtn) nextBtn.addEventListener("click", () => selectTrack(idx + 1, true));
  rows.forEach((row) => {{
    row.addEventListener("click", () => {{
      const target = Number(row.dataset.idx);
      if (target === idx) togglePlay();
      else selectTrack(target, true);
    }});
  }});

  volBtn.addEventListener("click", () => {{
    audio.muted = !audio.muted;
    volBtn.innerHTML = audio.muted ? MUTE_SVG : VOL_SVG;
  }});

  seek.addEventListener("input", () => {{
    seeking = true;
    const pct = Number(seek.value) / 10;
    seek.style.setProperty("--pct", `${{pct}}%`);
    const dur = Number.isFinite(audio.duration) && audio.duration > 0 ? audio.duration : (TRACKS[idx]?.durationSec || 0);
    timeEl.textContent = `${{fmt((pct / 100) * dur)}} / ${{fmt(dur)}}`;
  }});
  seek.addEventListener("change", () => {{
    const dur = Number.isFinite(audio.duration) && audio.duration > 0 ? audio.duration : 0;
    if (dur > 0) audio.currentTime = (Number(seek.value) / 1000) * dur;
    seeking = false;
  }});

  audio.addEventListener("timeupdate", () => {{
    if (seeking) return;
    const dur = Number.isFinite(audio.duration) && audio.duration > 0 ? audio.duration : (TRACKS[idx]?.durationSec || 0);
    const cur = audio.currentTime || 0;
    const pct = dur > 0 ? Math.min(100, Math.max(0, (cur / dur) * 100)) : 0;
    seek.value = String(Math.round(pct * 10));
    seek.style.setProperty("--pct", `${{pct.toFixed(1)}}%`);
    timeEl.textContent = `${{fmt(cur)}} / ${{fmt(dur)}}`;
  }});
  audio.addEventListener("play", renderActiveRow);
  audio.addEventListener("pause", renderActiveRow);
  audio.addEventListener("ended", () => {{
    if (idx + 1 < TRACKS.length) selectTrack(idx + 1, true);
    else renderActiveRow();
  }});
  audio.addEventListener("error", async () => {{
    const t = TRACKS[idx];
    if (t && !retriedRefresh) {{
      retriedRefresh = true;
      t.expiresAt = 0;
      try {{
        const fresh = await ensureFreshUrl(t);
        audio.src = fresh;
        await audio.play();
        return;
      }} catch (_err) {{}}
    }}
    showErr("Поток временно недоступен");
    renderActiveRow();
  }});

  selectTrack(0, false);
}})();
</script>
</body>
</html>"""
    return HTMLResponse(content=doc, headers=_headers(rec.allowed_origins))


@router.get("/{widget_id}/open")
async def open_widget_in_kwhale(widget_id: str, request: Request) -> Response:
    try:
        rec = await embed_service.get_public_widget(widget_id)
    except embed_service.EmbedInactiveError:
        return _render_embed_status("Интеграция отключена", "Виджет отозван администратором KWhale.", 410)
    except embed_service.EmbedNotFoundError:
        return _render_embed_status("Карточка не найдена", "Проверьте ссылку embed-интеграции.", 404)

    share = await share_service.create_or_refresh_share(rec.kind, rec.entity_id, user_id=None)
    origin = share_service.request_origin(request)
    return RedirectResponse(
        url=f"{origin}/share/s/{share.id}",
        status_code=302,
        headers={"Cache-Control": "no-store"},
    )


@router.get("/{widget_id}/stream/{track_id}")
async def refresh_widget_track_stream(widget_id: str, track_id: int) -> JSONResponse:
    try:
        payload = await embed_service.get_fresh_track_stream(widget_id, track_id)
    except embed_service.EmbedInactiveError as exc:
        raise ProblemError(410, "embed_disabled", "Интеграция отключена") from exc
    except embed_service.EmbedNotFoundError as exc:
        raise ProblemError(404, "not_found", "Трек интеграции не найден") from exc
    return JSONResponse(content=payload, headers={"Cache-Control": "no-store"})
