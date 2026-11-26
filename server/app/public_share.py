"""Unauthenticated Material 3 Web preview & OG card routes (`/share/*`)."""

from __future__ import annotations

import hashlib
import html
import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse

from .config import settings
from .services import share_card, share_service
from .v1 import images

router = APIRouter(prefix="/share", tags=["share-public"])

_STATIC_DIR = (Path(__file__).resolve().parent / "static" / "share").resolve()

_MIME_TYPES: dict[str, str] = {
    ".js": "application/javascript; charset=utf-8",
    ".svg": "image/svg+xml; charset=utf-8",
    ".png": "image/png",
    ".woff2": "font/woff2",
}

_KIND_LABELS: dict[str, str] = {
    "track": "Трек",
    "album": "Альбом",
    "artist": "Исполнитель",
}

_OG_TYPES: dict[str, str] = {
    "track": "music.song",
    "album": "music.album",
    "artist": "profile",
}

_COPY_SVG = (
    '<svg viewBox="0 0 24 24" aria-hidden="true">'
    '<path d="M16 1H4c-1.1 0-2 .9-2 2v14h2V3h12V1zm3 4H8c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h11'
    'c1.1 0 2-.9 2-2V7c0-1.1-.9-2-2-2zm0 16H8V7h11v14z" fill="currentColor"/>'
    "</svg>"
)

_SHARE_SVG = (
    '<svg viewBox="0 0 24 24" aria-hidden="true">'
    '<path d="M18 16.08c-.76 0-1.44.3-1.96.77L8.91 12.7c.05-.23.09-.46.09-.7s-.04-.47-.09-.7l7.05'
    '-4.11c.54.5 1.25.81 2.04.81 1.66 0 3-1.34 3-3s-1.34-3-3-3-3 1.34-3 3c0 .24.04.47.09.7L8.04'
    ' 9.81C7.5 9.31 6.79 9 6 9c-1.66 0-3 1.34-3 3s1.34 3 3 3c.79 0 1.5-.31 2.04-.81l7.12 4.16'
    'c-.05.21-.08.43-.08.65 0 1.61 1.31 2.92 2.92 2.92 1.61 0 2.92-1.31 2.92-2.92s-1.31-2.92-2.92-2.92z" '
    'fill="currentColor"/>'
    "</svg>"
)

_OPEN_EXT_SVG = (
    '<svg viewBox="0 0 24 24" aria-hidden="true">'
    '<path d="M19 19H5V5h7V3H5a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14c1.1 0 2-.9 2-2v-7h-2v7z'
    'M14 3v2h3.59l-9.83 9.83 1.41 1.41L19 6.41V10h2V3h-7z" fill="currentColor"/>'
    "</svg>"
)

_DEFAULT_PLATFORM_SVG = (
    '<svg viewBox="0 0 144 144" aria-hidden="true">'
    '<rect width="144" height="144" rx="41" fill="currentColor" fill-opacity="0.12"/>'
    '<path d="M72 34v52.7a18 18 0 1 0 9 15.6V52h18V34H72z" fill="currentColor"/>'
    "</svg>"
)


@lru_cache(maxsize=16)
def _load_platform_svg(platform: str) -> str:
    if not re.fullmatch(r"[a-z0-9_-]{1,32}", platform):
        return _DEFAULT_PLATFORM_SVG
    svg_path = (_STATIC_DIR / "icons" / f"{platform}.svg").resolve()
    if not svg_path.is_relative_to(_STATIC_DIR) or not svg_path.is_file():
        return _DEFAULT_PLATFORM_SVG
    raw = svg_path.read_text(encoding="utf-8", errors="ignore").strip()
    raw = re.sub(r"<\?xml[^>]*\?>\s*", "", raw)
    if "<svg" not in raw:
        return _DEFAULT_PLATFORM_SVG
    return re.sub(r"<svg\b", '<svg class="dest-icon-svg" aria-hidden="true"', raw, count=1)


def _token_css(tokens: dict[str, dict[str, Any]]) -> str:
    dark_vars = "; ".join(f"{k}: {v}" for k, v in tokens["dark"]["css"].items())
    light_vars = "; ".join(f"{k}: {v}" for k, v in tokens["light"]["css"].items())
    return (
        f":root {{ color-scheme: dark light; {dark_vars}; }}\n"
        f"@media (prefers-color-scheme: light) {{ :root {{ {light_vars}; }} }}"
    )


def _head_assets() -> str:
    return (
        '<link rel="icon" type="image/png" href="/share/static/favicon.png">\n'
        '<link rel="apple-touch-icon" href="/share/static/kwhale-logo.png">\n'
        '<script type="module" src="/share/static/m3-share.js"></script>'
    )


FONT_FACE_CSS = """
@font-face {
  font-family: 'Roboto';
  font-style: normal;
  font-weight: 400;
  font-display: swap;
  src: url('/share/static/fonts/roboto-cyrillic-400-normal.woff2') format('woff2');
  unicode-range: U+0301,U+0400-045F,U+0490-0491,U+04B0-04B1,U+2116;
}
@font-face {
  font-family: 'Roboto';
  font-style: normal;
  font-weight: 400;
  font-display: swap;
  src: url('/share/static/fonts/roboto-latin-400-normal.woff2') format('woff2');
  unicode-range: U+0000-00FF,U+0131,U+0152-0153,U+02BB-02BC,U+02C6,U+02DA,U+02DC,U+0304,U+0308,U+0329,U+2000-206F,U+20AC,U+2122,U+2191,U+2193,U+2212,U+2215,U+FEFF,U+FFFD;
}
@font-face {
  font-family: 'Roboto';
  font-style: normal;
  font-weight: 500;
  font-display: swap;
  src: url('/share/static/fonts/roboto-cyrillic-500-normal.woff2') format('woff2');
  unicode-range: U+0301,U+0400-045F,U+0490-0491,U+04B0-04B1,U+2116;
}
@font-face {
  font-family: 'Roboto';
  font-style: normal;
  font-weight: 500;
  font-display: swap;
  src: url('/share/static/fonts/roboto-latin-500-normal.woff2') format('woff2');
  unicode-range: U+0000-00FF,U+0131,U+0152-0153,U+02BB-02BC,U+02C6,U+02DA,U+02DC,U+0304,U+0308,U+0329,U+2000-206F,U+20AC,U+2122,U+2191,U+2193,U+2212,U+2215,U+FEFF,U+FFFD;
}
@font-face {
  font-family: 'Roboto';
  font-style: normal;
  font-weight: 700;
  font-display: swap;
  src: url('/share/static/fonts/roboto-cyrillic-700-normal.woff2') format('woff2');
  unicode-range: U+0301,U+0400-045F,U+0490-0491,U+04B0-04B1,U+2116;
}
@font-face {
  font-family: 'Roboto';
  font-style: normal;
  font-weight: 700;
  font-display: swap;
  src: url('/share/static/fonts/roboto-latin-700-normal.woff2') format('woff2');
  unicode-range: U+0000-00FF,U+0131,U+0152-0153,U+02BB-02BC,U+02C6,U+02DA,U+02DC,U+0304,U+0308,U+0329,U+2000-206F,U+20AC,U+2122,U+2191,U+2193,U+2212,U+2215,U+FEFF,U+FFFD;
}
""".strip()


def _base_styles(token_css: str) -> str:
    return f"""
<style>
{FONT_FACE_CSS}
{token_css}
:root {{
  --md-sys-shape-corner-full: 9999px;
  --md-sys-typescale-body-large-font: 'Roboto', system-ui, -apple-system, 'Segoe UI', sans-serif;
  --md-sys-typescale-body-large-size: 16px;
  --md-sys-typescale-body-large-line-height: 24px;
  --md-sys-typescale-body-large-weight: 500;
  --md-sys-typescale-body-medium-font: 'Roboto', system-ui, -apple-system, 'Segoe UI', sans-serif;
  --md-sys-typescale-body-medium-size: 14px;
  --md-sys-typescale-body-medium-line-height: 20px;
  --md-sys-typescale-body-medium-weight: 400;
  --md-sys-typescale-label-large-font: 'Roboto', system-ui, -apple-system, 'Segoe UI', sans-serif;
  --md-sys-typescale-label-large-size: 14px;
  --md-sys-typescale-label-large-line-height: 20px;
  --md-sys-typescale-label-large-weight: 500;
  --md-sys-typescale-label-small-font: 'Roboto', system-ui, -apple-system, 'Segoe UI', sans-serif;
  --md-sys-typescale-label-small-size: 11px;
  --md-sys-typescale-label-small-line-height: 16px;
  --md-sys-typescale-label-small-weight: 500;
  --md-filled-button-container-height: 48px;
  --md-filled-button-leading-space: 24px;
  --md-filled-button-trailing-space: 24px;
  --md-filled-button-with-leading-icon-leading-space: 18px;
  --md-filled-button-with-leading-icon-trailing-space: 24px;
  --md-filled-button-icon-size: 20px;
  --md-filled-tonal-button-container-height: 48px;
  --md-filled-tonal-button-leading-space: 22px;
  --md-filled-tonal-button-trailing-space: 22px;
  --md-filled-tonal-button-with-leading-icon-leading-space: 18px;
  --md-filled-tonal-button-with-leading-icon-trailing-space: 22px;
  --md-filled-tonal-button-icon-size: 18px;
  --md-list-container-color: transparent;
  --md-list-item-container-color: transparent;
  --md-list-item-two-line-container-height: 70px;
  --md-list-item-leading-space: 18px;
  --md-list-item-trailing-space: 18px;
  --md-list-item-label-text-weight: 500;
  --md-divider-color: var(--md-sys-color-outline-variant);
}}
*, *::before, *::after {{ box-sizing: border-box; }}
html, body {{ margin: 0; padding: 0; min-height: 100%; }}
body {{
  background-color: var(--md-sys-color-surface);
  background-image: radial-gradient(
    ellipse 90% 420px at 50% 0%,
    color-mix(in srgb, var(--md-sys-color-primary-container) 34%, transparent) 0%,
    transparent 100%
  );
  background-repeat: no-repeat;
  color: var(--md-sys-color-on-surface);
  font-family: 'Roboto', system-ui, -apple-system, 'Segoe UI', sans-serif;
  line-height: 1.45;
  -webkit-font-smoothing: antialiased;
}}
.shell {{
  width: 100%;
  max-width: 520px;
  margin: 0 auto;
  padding: max(8px, env(safe-area-inset-top)) 20px max(36px, env(safe-area-inset-bottom));
}}
/* M3 Center-Aligned Top App Bar (64dp) */
.topbar {{
  height: 64px;
  display: grid;
  grid-template-columns: 44px 1fr 44px;
  align-items: center;
  margin-bottom: 12px;
}}
.brand {{
  grid-column: 2;
  justify-self: center;
  display: inline-flex;
  align-items: center;
  gap: 10px;
  text-decoration: none;
  color: var(--md-sys-color-on-surface);
  font-weight: 500;
  font-size: 18px;
  letter-spacing: -0.01em;
}}
.brand-logo {{
  width: 30px;
  height: 30px;
  border-radius: 9px;
  display: block;
  flex-shrink: 0;
}}
.topbar-action {{
  grid-column: 3;
  justify-self: end;
}}
.topbar-action svg {{
  width: 22px;
  height: 22px;
  display: block;
}}
/* Centered Hero */
.hero {{
  display: flex;
  flex-direction: column;
  align-items: center;
  text-align: center;
  padding: 8px 0 28px;
}}
.art-frame {{
  position: relative;
  width: min(256px, 66vw);
  aspect-ratio: 1 / 1;
  border-radius: 28px;
  background: var(--md-sys-color-surface-container-high);
  border: 1px solid color-mix(in srgb, var(--md-sys-color-outline-variant) 72%, transparent);
  margin-bottom: 24px;
  --md-elevation-level: 2;
}}
.art-frame.is-artist {{
  border-radius: 9999px;
}}
.art-img {{
  width: 100%;
  height: 100%;
  object-fit: cover;
  display: block;
  border-radius: inherit;
}}
h1 {{
  margin: 0;
  font-size: clamp(26px, 6vw, 32px);
  line-height: 1.18;
  font-weight: 700;
  letter-spacing: -0.02em;
  color: var(--md-sys-color-on-surface);
  max-width: 100%;
  word-break: break-word;
}}
.subtitle {{
  margin: 8px 0 0;
  font-size: 17px;
  line-height: 24px;
  font-weight: 500;
  color: var(--md-sys-color-primary);
}}
.meta {{
  margin: 6px 0 0;
  font-size: 14px;
  line-height: 20px;
  color: var(--md-sys-color-on-surface-variant);
}}
.actions {{
  display: flex;
  flex-wrap: wrap;
  justify-content: center;
  gap: 12px;
  margin-top: 24px;
  width: 100%;
}}
.btn-orca {{
  width: 20px;
  height: 20px;
  display: block;
  background-color: currentColor;
  -webkit-mask: url('/share/static/kwhale-mark.png') center / contain no-repeat;
  mask: url('/share/static/kwhale-mark.png') center / contain no-repeat;
}}
.btn-icon-svg {{
  width: 18px;
  height: 18px;
  display: block;
}}
/* M3 Destinations Card & List */
.dest-card {{
  position: relative;
  border-radius: 28px;
  background: var(--md-sys-color-surface-container-low);
  border: 1px solid color-mix(in srgb, var(--md-sys-color-outline-variant) 68%, transparent);
  overflow: hidden;
  --md-elevation-level: 1;
}}
.dest-heading {{
  margin: 0;
  padding: 18px 20px 6px;
  font-size: 13px;
  line-height: 18px;
  font-weight: 500;
  color: var(--md-sys-color-on-surface-variant);
}}
md-list {{
  padding: 4px 0 10px;
}}
.dest-icon-wrap {{
  width: 40px;
  height: 40px;
  display: flex;
  align-items: center;
  justify-content: center;
}}
.dest-icon-svg {{
  width: 40px;
  height: 40px;
  display: block;
}}
.dest-end-svg {{
  width: 20px;
  height: 20px;
  color: var(--md-sys-color-on-surface-variant);
  opacity: 0.72;
  display: block;
}}
.support-exact {{
  color: var(--md-sys-color-primary);
}}
.empty-dest {{
  margin: 0;
  padding: 16px 20px 22px;
  font-size: 14px;
  color: var(--md-sys-color-on-surface-variant);
}}
.status-card {{
  border-radius: 28px;
  background: var(--md-sys-color-surface-container-low);
  border: 1px solid color-mix(in srgb, var(--md-sys-color-outline-variant) 68%, transparent);
  padding: 32px 24px;
  text-align: center;
  display: flex;
  flex-direction: column;
  align-items: center;
}}
.status-logo {{
  width: 56px;
  height: 56px;
  border-radius: 16px;
  margin-bottom: 18px;
  display: block;
}}
.footnote {{
  margin: 16px 8px 0;
  text-align: center;
  font-size: 12px;
  line-height: 18px;
  color: var(--md-sys-color-on-surface-variant);
}}
.snackbar {{
  position: fixed;
  left: 50%;
  bottom: max(20px, env(safe-area-inset-bottom));
  transform: translate(-50%, 16px);
  opacity: 0;
  pointer-events: none;
  background: var(--md-sys-color-on-surface);
  color: var(--md-sys-color-surface);
  padding: 12px 20px;
  border-radius: 14px;
  font-size: 14px;
  font-weight: 500;
  transition: opacity 160ms ease, transform 180ms cubic-bezier(0.2, 0, 0, 1);
  z-index: 20;
  max-width: calc(100vw - 32px);
  text-align: center;
}}
.snackbar.show {{
  opacity: 1;
  transform: translate(-50%, 0);
}}
/* SSR / pre-hydration fallbacks so first paint has zero layout jump */
md-filled-button:not(:defined),
md-filled-tonal-button:not(:defined) {{
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
  height: 48px;
  padding: 0 22px;
  border-radius: 9999px;
  font-size: 14px;
  font-weight: 500;
  text-decoration: none;
  cursor: pointer;
  white-space: nowrap;
}}
md-filled-button:not(:defined) {{
  background: var(--md-sys-color-primary);
  color: var(--md-sys-color-on-primary);
}}
md-filled-tonal-button:not(:defined) {{
  background: var(--md-sys-color-secondary-container);
  color: var(--md-sys-color-on-secondary-container);
}}
md-icon-button:not(:defined) {{
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 40px;
  height: 40px;
  border-radius: 9999px;
  color: var(--md-sys-color-on-surface-variant);
  cursor: pointer;
}}
md-list:not(:defined) {{
  display: flex;
  flex-direction: column;
}}
md-list-item:not(:defined) {{
  display: flex;
  align-items: center;
  gap: 16px;
  min-height: 70px;
  padding: 12px 18px;
  color: var(--md-sys-color-on-surface);
  cursor: pointer;
}}
md-list-item:not(:defined) [slot="headline"] {{
  display: block;
  font-size: 16px;
  font-weight: 500;
}}
md-list-item:not(:defined) [slot="supporting-text"] {{
  display: block;
  font-size: 14px;
  color: var(--md-sys-color-on-surface-variant);
}}
md-list-item:not(:defined) [slot="end"] {{
  margin-left: auto;
}}
md-divider:not(:defined) {{
  display: block;
  height: 1px;
  background: var(--md-sys-color-outline-variant);
  margin-left: 16px;
}}
@media (max-width: 420px) {{
  .actions md-filled-button,
  .actions md-filled-tonal-button {{
    width: 100%;
  }}
}}
@media (prefers-reduced-motion: reduce) {{
  .snackbar {{ transition: none !important; }}
}}
</style>
"""


def _format_meta_line(rec: share_service.ShareRecord) -> str:
    kind_label = _KIND_LABELS.get(rec.kind, "Релиз")
    parts = [kind_label]
    if rec.meta:
        parts.append(rec.meta)
    elif rec.kind == "artist":
        parts.append("KWhale")
    return " · ".join(parts)


def _render_status_page(title: str, body: str, status_code: int) -> HTMLResponse:
    seed = share_card.PaletteSeed(hue_deg=214.0, saturation=0.45)
    tokens = share_card.build_monet_tokens(seed)
    css = _base_styles(_token_css(tokens))
    doc = f"""<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<title>{html.escape(title)} — KWhale</title>
{_head_assets()}
{css}
</head>
<body>
<main class="shell">
  <header class="topbar">
    <span></span>
    <a class="brand" href="/share/">
      <img class="brand-logo" src="/share/static/kwhale-logo.png" width="30" height="30" alt="">
      <span>KWhale</span>
    </a>
    <span></span>
  </header>
  <section class="status-card">
    <img class="status-logo" src="/share/static/kwhale-logo.png" width="56" height="56" alt="">
    <h1>{html.escape(title)}</h1>
    <p class="meta" style="margin-top:10px;max-width:38ch">{html.escape(body)}</p>
  </section>
</main>
</body>
</html>"""
    return HTMLResponse(content=doc, status_code=status_code, headers={"Cache-Control": "no-store"})


@router.get("/static/{asset_path:path}", include_in_schema=False)
async def share_static_asset(asset_path: str, request: Request) -> Response:
    """Serve bundled M3 Web components, Roboto WOFF2 subsets, official brand SVGs & logos."""
    target = (_STATIC_DIR / asset_path).resolve()
    if not target.is_relative_to(_STATIC_DIR) or not target.is_file():
        return Response(status_code=404, headers={"Cache-Control": "no-store"})
    media_type = _MIME_TYPES.get(target.suffix.lower())
    if media_type is None:
        return Response(status_code=404, headers={"Cache-Control": "no-store"})
    data = target.read_bytes()
    etag = hashlib.sha256(data).hexdigest()[:16]
    headers = {"Cache-Control": "public, max-age=86400", "ETag": f'"{etag}"'}
    if request.headers.get("if-none-match", "").strip('" ') == etag:
        return Response(status_code=304, headers=headers)
    return Response(content=data, media_type=media_type, headers=headers)


@router.get("", include_in_schema=False)
@router.get("/", response_class=HTMLResponse)
async def share_landing() -> HTMLResponse:
    """Clean centered M3 public landing page without exposing private library catalog."""
    seed = share_card.PaletteSeed(hue_deg=214.0, saturation=0.52)
    tokens = share_card.build_monet_tokens(seed)
    css = _base_styles(_token_css(tokens))
    doc = f"""<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<title>KWhale Share</title>
<meta name="description" content="Обмен треками, альбомами и исполнителями из KWhale с прямым переходом в приложение или внешние стриминги.">
{_head_assets()}
{css}
</head>
<body>
<main class="shell">
  <header class="topbar">
    <span></span>
    <div class="brand">
      <img class="brand-logo" src="/share/static/kwhale-logo.png" width="30" height="30" alt="">
      <span>KWhale</span>
    </div>
    <span></span>
  </header>
  <section class="status-card">
    <img class="status-logo" src="/share/static/kwhale-logo.png" width="56" height="56" alt="">
    <h1>Музыкальные ссылки KWhale</h1>
    <p class="meta" style="margin-top:12px;max-width:40ch">
      Откройте ссылку формата <code>/share/s/&lt;id&gt;</code>, отправленную из приложения KWhale,
      чтобы послушать трек, альбом или исполнителя в KWhale или перейти на подключённые сервисы.
    </p>
  </section>
</main>
</body>
</html>"""
    return HTMLResponse(content=doc, headers={"Cache-Control": "public, max-age=300"})


@router.get("/health")
async def share_health() -> JSONResponse:
    return JSONResponse(
        content={"status": "ok", "service": "kwhale-share", "ttlDays": settings.share_link_ttl_days},
        headers={"Cache-Control": "no-store"},
    )


@router.get("/s/{share_id}", response_class=HTMLResponse)
async def share_page(share_id: str, request: Request) -> HTMLResponse:
    try:
        rec = await share_service.get_share(share_id)
    except share_service.ShareExpiredError:
        return _render_status_page(
            "Ссылка устарела",
            f"Срок действия короткой ссылки ({settings.share_link_ttl_days} дн.) истёк. Попросите отправить её ещё раз из KWhale.",
            410,
        )
    except share_service.ShareNotFoundError:
        return _render_status_page(
            "Ссылка не найдена",
            "Проверьте адрес или запросите актуальную ссылку из приложения KWhale.",
            404,
        )

    origin = share_service.request_origin(request)
    share_url = f"{origin}/share/s/{rec.id}"
    card_url = f"{origin}/share/card/{rec.id}.png"
    cover_url = f"{origin}/share/cover/{rec.id}"
    deep_link = share_service.deep_link_for(rec.kind, rec.entity_id)
    intent_uri = share_service.android_intent_uri(rec.kind, rec.entity_id, share_url)

    tokens = await share_card.get_monet_tokens(rec.image_id, rec.subtitle or rec.title)
    css = _base_styles(_token_css(tokens))

    og_type = _OG_TYPES.get(rec.kind, "website")
    og_title = rec.title if not rec.subtitle or rec.kind == "artist" else f"{rec.subtitle} — {rec.title}"
    meta_line = _format_meta_line(rec)
    desc_parts = [p for p in (meta_line, " · ".join(p.label for p in rec.platforms[:4])) if p]
    og_desc = " · ".join(desc_parts) or "Слушать в KWhale или на любимой площадке"
    expires_label = rec.expires_at.astimezone().strftime("%d.%m.%Y")

    list_nodes: list[str] = []
    for idx, p in enumerate(rec.platforms):
        icon_svg = _load_platform_svg(p.platform)
        support_cls = "support-exact" if p.exact else ""
        status_label = "Точная ссылка" if p.exact else "Поиск по названию"
        go_href = html.escape(f"/share/go/{rec.id}/{p.platform}", quote=True)
        if idx > 0:
            list_nodes.append('<md-divider inset></md-divider>')
        list_nodes.append(
            f'<md-list-item type="link" href="{go_href}">'
            f'<span slot="start" class="dest-icon-wrap">{icon_svg}</span>'
            f'<div slot="headline">{html.escape(p.label)}</div>'
            f'<div slot="supporting-text" class="{support_cls}">{html.escape(status_label)}</div>'
            f'<span slot="end" class="dest-end-svg">{_OPEN_EXT_SVG}</span>'
            f"</md-list-item>"
        )

    destinations_body = (
        f'<md-list>{"".join(list_nodes)}</md-list>'
        if list_nodes
        else '<p class="empty-dest">Прямые ссылки на внешние площадки для этого релиза пока не найдены.</p>'
    )

    art_class = "art-frame is-artist" if rec.kind == "artist" else "art-frame"
    sub_html = (
        f'<p class="subtitle">{html.escape(rec.subtitle)}</p>'
        if rec.subtitle and (rec.kind != "artist" or rec.subtitle != "Исполнитель")
        else ""
    )

    doc = f"""<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<title>{html.escape(og_title)} — KWhale</title>
<meta name="description" content="{html.escape(og_desc, quote=True)}">
<link rel="canonical" href="{html.escape(share_url, quote=True)}">
<link rel="alternate" href="{html.escape(deep_link, quote=True)}">
<meta property="og:site_name" content="KWhale">
<meta property="og:type" content="{html.escape(og_type, quote=True)}">
<meta property="og:title" content="{html.escape(og_title, quote=True)}">
<meta property="og:description" content="{html.escape(og_desc, quote=True)}">
<meta property="og:url" content="{html.escape(share_url, quote=True)}">
<meta property="og:image" content="{html.escape(card_url, quote=True)}">
<meta property="og:image:type" content="image/png">
<meta property="og:image:width" content="1200">
<meta property="og:image:height" content="630">
<meta name="twitter:card" content="summary_large_image">
<meta name="twitter:title" content="{html.escape(og_title, quote=True)}">
<meta name="twitter:description" content="{html.escape(og_desc, quote=True)}">
<meta name="twitter:image" content="{html.escape(card_url, quote=True)}">
{_head_assets()}
{css}
</head>
<body>
<main class="shell">
  <header class="topbar">
    <span></span>
    <a class="brand" href="/share/">
      <img class="brand-logo" src="/share/static/kwhale-logo.png" width="30" height="30" alt="">
      <span>KWhale</span>
    </a>
    <md-icon-button id="top-share" class="topbar-action" type="button" aria-label="Поделиться ссылкой">
      {_SHARE_SVG}
    </md-icon-button>
  </header>

  <section class="hero" aria-label="Карточка релиза">
    <div class="{art_class}">
      <md-elevation></md-elevation>
      <img class="art-img" src="{html.escape(cover_url, quote=True)}" alt="{html.escape(og_title, quote=True)}" loading="eager">
    </div>
    <h1>{html.escape(rec.title)}</h1>
    {sub_html}
    <p class="meta">{html.escape(meta_line)}</p>
    <div class="actions">
      <md-filled-button id="open-app" href="{html.escape(deep_link, quote=True)}">
        <span slot="icon" class="btn-orca" aria-hidden="true"></span>
        Открыть в KWhale
      </md-filled-button>
      <md-filled-tonal-button id="copy-link" type="button">
        <span slot="icon" class="btn-icon-svg" aria-hidden="true">{_COPY_SVG}</span>
        <span id="copy-label">Скопировать ссылку</span>
      </md-filled-tonal-button>
    </div>
  </section>

  <section aria-label="Стриминговые площадки">
    <div class="dest-card">
      <md-elevation></md-elevation>
      <h2 class="dest-heading">Слушать на площадках</h2>
      {destinations_body}
    </div>
    <p class="footnote">Ссылка активна до {html.escape(expires_label)}</p>
  </section>
</main>
<div id="snackbar" class="snackbar" role="status" aria-live="polite"></div>
<script>
(() => {{
  const shareUrl = {json.dumps(share_url)};
  const shareTitle = {json.dumps(og_title)};
  const deepLink = {json.dumps(deep_link)};
  const intentUri = {json.dumps(intent_uri)};
  const snackbar = document.getElementById("snackbar");
  const openBtn = document.getElementById("open-app");
  const copyBtn = document.getElementById("copy-link");
  const copyLbl = document.getElementById("copy-label");
  const topShare = document.getElementById("top-share");
  let timer = null;

  function showToast(msg) {{
    snackbar.textContent = msg;
    snackbar.classList.add("show");
    clearTimeout(timer);
    timer = setTimeout(() => snackbar.classList.remove("show"), 2600);
  }}

  async function copyShareUrl() {{
    try {{
      await navigator.clipboard.writeText(shareUrl);
    }} catch (_err) {{
      const ta = document.createElement("textarea");
      ta.value = shareUrl;
      ta.style.position = "fixed";
      ta.style.opacity = "0";
      document.body.appendChild(ta);
      ta.select();
      document.execCommand("copy");
      document.body.removeChild(ta);
    }}
    copyLbl.textContent = "Скопировано";
    showToast("Ссылка скопирована");
    setTimeout(() => {{ copyLbl.textContent = "Скопировать ссылку"; }}, 1800);
  }}

  openBtn.addEventListener("click", (ev) => {{
    const ua = navigator.userAgent || "";
    const isAndroid = /Android/i.test(ua);
    if (!isAndroid) {{
      if (!window.customElements || !window.customElements.get("md-filled-button")) {{
        window.location.href = deepLink;
      }}
      return;
    }}
    ev.preventDefault();
    const isChrome = /Chrome\\//.test(ua) && !/wv\\)/.test(ua);
    window.location.href = isChrome ? intentUri : deepLink;
    setTimeout(() => {{
      if (!document.hidden) {{
        showToast("Если KWhale не установлен, выберите сервис из списка");
      }}
    }}, 1100);
  }});

  copyBtn.addEventListener("click", copyShareUrl);

  topShare.addEventListener("click", async () => {{
    if (navigator.share) {{
      try {{
        await navigator.share({{ title: shareTitle, url: shareUrl }});
        return;
      }} catch (_err) {{}}
    }}
    copyShareUrl();
  }});

  // Progressive fallback if custom elements have not upgraded yet
  document.addEventListener("click", (ev) => {{
    if (window.customElements && window.customElements.get("md-list-item")) return;
    const item = ev.target && ev.target.closest ? ev.target.closest("md-list-item[href]") : null;
    if (item) {{
      window.location.href = item.getAttribute("href");
    }}
  }});
}})();
</script>
</body>
</html>"""
    return HTMLResponse(
        content=doc,
        headers={
            "Cache-Control": "public, max-age=300, s-maxage=600",
            "Vary": "Accept-Encoding",
        },
    )


@router.get("/card/{share_id}.png")
@router.get("/card/{share_id}.jpg")
async def share_card_image(share_id: str, request: Request) -> Response:
    fmt = "jpg" if request.url.path.lower().endswith(".jpg") else "png"
    try:
        rec = await share_service.get_share(share_id)
    except share_service.ShareExpiredError:
        return Response(status_code=410, headers={"Cache-Control": "no-store"})
    except share_service.ShareNotFoundError:
        return Response(status_code=404, headers={"Cache-Control": "no-store"})

    raw, mime, etag = await share_card.render_share_card(
        rec, fmt=fmt, origin=share_service.request_origin(request)
    )
    inm = request.headers.get("if-none-match", "").strip('" ')
    headers = {"Cache-Control": "public, max-age=86400", "ETag": f'"{etag}"'}
    if inm == etag:
        return Response(status_code=304, headers=headers)
    return Response(content=raw, media_type=mime, headers=headers)


@router.get("/cover/{share_id}")
async def share_cover_image(share_id: str, request: Request) -> Response:
    try:
        rec = await share_service.get_share(share_id)
    except share_service.ShareExpiredError:
        return Response(status_code=410, headers={"Cache-Control": "no-store"})
    except share_service.ShareNotFoundError:
        return Response(status_code=404, headers={"Cache-Control": "no-store"})

    data, mime = await share_card.load_cover_bytes(rec.image_id)
    if data:
        etag = hashlib.sha256(data).hexdigest()[:32]
        inm = request.headers.get("if-none-match", "").strip('" ')
        headers = {"Cache-Control": "public, max-age=86400, immutable", "ETag": f'"{etag}"'}
        if inm == etag:
            return Response(status_code=304, headers=headers)
        return Response(content=data, media_type=mime, headers=headers)

    return await images.get_image(images.PLACEHOLDER_ID, size="md")


@router.get("/go/{share_id}/{platform}")
async def share_redirect(share_id: str, platform: str) -> Response:
    try:
        target = await share_service.record_click(share_id, platform)
    except share_service.ShareExpiredError:
        return _render_status_page("Ссылка устарела", "Срок действия ссылки истёк.", 410)
    except share_service.ShareNotFoundError:
        return _render_status_page("Переход недоступен", "Площадка или ссылка не найдена.", 404)

    return RedirectResponse(url=target, status_code=302, headers={"Cache-Control": "no-store"})
