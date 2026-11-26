"""Material You 3 Monet palette extractor and 1200x630 OG card renderer."""

from __future__ import annotations

import asyncio
import colorsys
import hashlib
import io
import os
from collections import OrderedDict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, cast

from PIL import Image, ImageDraw, ImageFilter, ImageFont

from ..v1 import images
from .share_service import ShareRecord

_CARD_W = 1200
_CARD_H = 630

_BOLD_FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/liberation-sans-fonts/LiberationSans-Bold.ttf",
)
_REG_FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/liberation-sans-fonts/LiberationSans-Regular.ttf",
)

_CARD_CACHE_MAX = 64
_PALETTE_CACHE_MAX = 128
_card_cache: OrderedDict[tuple[str, str], tuple[bytes, str, str]] = OrderedDict()
_palette_cache: OrderedDict[str, dict[str, Any]] = OrderedDict()


@dataclass(frozen=True, slots=True)
class PaletteSeed:
    hue_deg: float
    saturation: float


def _fnv1a_hue(seed: str) -> float:
    h = 0x811C9DC5
    for ch in seed or "kwhale":
        h ^= ord(ch)
        h = (h * 0x01000193) & 0xFFFFFFFF
    # Bias away from mud: KWhale signature blue-indigo range blended cleanly.
    return float(h % 360)


def _hls_to_rgb_int(hue_deg: float, lightness: float, saturation: float) -> tuple[int, int, int]:
    r, g, b = colorsys.hls_to_rgb(
        (hue_deg % 360.0) / 360.0,
        max(0.0, min(1.0, lightness)),
        max(0.0, min(1.0, saturation)),
    )
    return int(round(r * 255)), int(round(g * 255)), int(round(b * 255))


def _rgb_hex(rgb: tuple[int, int, int]) -> str:
    return "#{:02X}{:02X}{:02X}".format(*rgb)


def _rel_luminance(rgb: tuple[int, int, int]) -> float:
    def _lin(c: int) -> float:
        s = c / 255.0
        return s / 12.92 if s <= 0.03928 else ((s + 0.055) / 1.055) ** 2.4

    r, g, b = rgb
    return 0.2126 * _lin(r) + 0.7152 * _lin(g) + 0.0722 * _lin(b)


def _contrast_ratio(a: tuple[int, int, int], b: tuple[int, int, int]) -> float:
    la, lb = _rel_luminance(a), _rel_luminance(b)
    hi, lo = max(la, lb), min(la, lb)
    return (hi + 0.05) / (lo + 0.05)


def _extract_seed(image_bytes: bytes | None, fallback_seed: str) -> PaletteSeed:
    if not image_bytes:
        return PaletteSeed(hue_deg=_fnv1a_hue(fallback_seed), saturation=0.45)
    try:
        with Image.open(io.BytesIO(image_bytes)) as src:
            thumb = src.convert("RGB").resize((48, 48), Image.Resampling.BILINEAR)
        best_score = 0.0
        best_hue = None
        best_sat = 0.0
        hue_bins: list[float] = [0.0] * 36
        sat_acc: list[float] = [0.0] * 36
        count_acc: list[int] = [0] * 36
        get_pixels = getattr(thumb, "get_flattened_data", thumb.getdata)
        pixels = cast(list[tuple[int, int, int]], list(get_pixels()))
        for r, g, b in pixels:
            h, s, v = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)
            if v < 0.14 or v > 0.94 or s < 0.16:
                continue
            weight = (s**1.4) * (1.0 - abs(v - 0.56))
            bucket = int(h * 36) % 36
            hue_bins[bucket] += weight
            sat_acc[bucket] += s
            count_acc[bucket] += 1
            if weight > best_score:
                best_score = weight
                best_hue = h * 360.0
                best_sat = s
        if best_hue is not None and any(hue_bins):
            top_bucket = max(range(36), key=lambda idx: hue_bins[idx])
            avg_sat = sat_acc[top_bucket] / max(1, count_acc[top_bucket])
            hue = (top_bucket + 0.5) * 10.0
            return PaletteSeed(hue_deg=hue, saturation=max(0.28, min(0.82, max(avg_sat, best_sat))))
    except Exception:  # noqa: BLE001 — fallback to deterministic cover seed
        pass
    return PaletteSeed(hue_deg=_fnv1a_hue(fallback_seed), saturation=0.42)


def build_monet_tokens(seed: PaletteSeed) -> dict[str, dict[str, Any]]:
    """Build accessible Material You 3 dark and light tonal schemes from a cover seed."""
    h = seed.hue_deg
    chroma = max(0.32, min(0.76, seed.saturation))
    subtle_chroma = max(0.08, min(0.18, chroma * 0.28))

    # Dark scheme tokens
    d_surface = _hls_to_rgb_int(h, 0.072, subtle_chroma)
    d_surf_low = _hls_to_rgb_int(h, 0.105, subtle_chroma * 1.1)
    d_surf_cont = _hls_to_rgb_int(h, 0.135, subtle_chroma * 1.15)
    d_surf_high = _hls_to_rgb_int(h, 0.18, subtle_chroma * 1.2)
    d_primary = _hls_to_rgb_int(h, 0.78, min(0.78, chroma + 0.12))
    d_on_primary = _hls_to_rgb_int(h, 0.14, min(0.55, chroma))
    if _contrast_ratio(d_primary, d_on_primary) < 4.5:
        d_primary = _hls_to_rgb_int(h, 0.84, min(0.75, chroma))
        d_on_primary = _hls_to_rgb_int(h, 0.10, 0.45)
    d_prim_cont = _hls_to_rgb_int(h, 0.25, min(0.52, chroma * 0.85))
    d_on_prim_cont = _hls_to_rgb_int(h, 0.90, min(0.60, chroma * 0.85))
    if _contrast_ratio(d_prim_cont, d_on_prim_cont) < 4.5:
        d_on_prim_cont = (238, 240, 246)
    d_on_surface = (232, 229, 236)
    d_on_surf_var = (196, 194, 203)
    d_outline = _hls_to_rgb_int(h, 0.30, subtle_chroma)
    d_secondary = _hls_to_rgb_int(h, 0.72, min(0.32, subtle_chroma * 2.2))
    d_sec_cont = _hls_to_rgb_int(h, 0.24, min(0.30, subtle_chroma * 2.0))
    d_on_sec_cont = _hls_to_rgb_int(h, 0.90, min(0.34, subtle_chroma * 2.2))
    if _contrast_ratio(d_sec_cont, d_on_sec_cont) < 4.5:
        d_on_sec_cont = (234, 236, 244)

    # Light scheme tokens
    l_surface = _hls_to_rgb_int(h, 0.975, min(0.28, subtle_chroma * 1.8))
    l_surf_low = _hls_to_rgb_int(h, 0.952, min(0.30, subtle_chroma * 1.9))
    l_surf_cont = _hls_to_rgb_int(h, 0.930, min(0.30, subtle_chroma * 2.0))
    l_surf_high = _hls_to_rgb_int(h, 0.900, min(0.30, subtle_chroma * 2.0))
    l_primary = _hls_to_rgb_int(h, 0.37, min(0.78, chroma + 0.10))
    l_on_primary = (255, 255, 255)
    if _contrast_ratio(l_primary, l_on_primary) < 4.5:
        l_primary = _hls_to_rgb_int(h, 0.30, min(0.75, chroma + 0.08))
    l_prim_cont = _hls_to_rgb_int(h, 0.89, min(0.65, chroma * 0.95))
    l_on_prim_cont = _hls_to_rgb_int(h, 0.14, min(0.65, chroma * 0.95))
    l_on_surface = (27, 27, 31)
    l_on_surf_var = (69, 70, 79)
    l_outline = _hls_to_rgb_int(h, 0.78, min(0.24, subtle_chroma * 1.5))
    l_secondary = _hls_to_rgb_int(h, 0.38, min(0.30, subtle_chroma * 2.0))
    l_sec_cont = _hls_to_rgb_int(h, 0.89, min(0.32, subtle_chroma * 2.1))
    l_on_sec_cont = _hls_to_rgb_int(h, 0.15, min(0.34, subtle_chroma * 2.2))
    if _contrast_ratio(l_sec_cont, l_on_sec_cont) < 4.5:
        l_on_sec_cont = (27, 27, 31)

    return {
        "dark": {
            "surface": d_surface,
            "surface_low": d_surf_low,
            "surface_container": d_surf_cont,
            "surface_high": d_surf_high,
            "primary": d_primary,
            "on_primary": d_on_primary,
            "primary_container": d_prim_cont,
            "on_primary_container": d_on_prim_cont,
            "secondary": d_secondary,
            "secondary_container": d_sec_cont,
            "on_secondary_container": d_on_sec_cont,
            "on_surface": d_on_surface,
            "on_surface_variant": d_on_surf_var,
            "outline_variant": d_outline,
            "css": {
                "--md-sys-color-surface": _rgb_hex(d_surface),
                "--md-sys-color-surface-container-low": _rgb_hex(d_surf_low),
                "--md-sys-color-surface-container": _rgb_hex(d_surf_cont),
                "--md-sys-color-surface-container-high": _rgb_hex(d_surf_high),
                "--md-sys-color-primary": _rgb_hex(d_primary),
                "--md-sys-color-on-primary": _rgb_hex(d_on_primary),
                "--md-sys-color-primary-container": _rgb_hex(d_prim_cont),
                "--md-sys-color-on-primary-container": _rgb_hex(d_on_prim_cont),
                "--md-sys-color-secondary": _rgb_hex(d_secondary),
                "--md-sys-color-secondary-container": _rgb_hex(d_sec_cont),
                "--md-sys-color-on-secondary-container": _rgb_hex(d_on_sec_cont),
                "--md-sys-color-on-surface": _rgb_hex(d_on_surface),
                "--md-sys-color-on-surface-variant": _rgb_hex(d_on_surf_var),
                "--md-sys-color-outline-variant": _rgb_hex(d_outline),
                "--md-sys-color-shadow": "#000000",
            },
        },
        "light": {
            "surface": l_surface,
            "surface_low": l_surf_low,
            "surface_container": l_surf_cont,
            "surface_high": l_surf_high,
            "primary": l_primary,
            "on_primary": l_on_primary,
            "primary_container": l_prim_cont,
            "on_primary_container": l_on_prim_cont,
            "secondary": l_secondary,
            "secondary_container": l_sec_cont,
            "on_secondary_container": l_on_sec_cont,
            "on_surface": l_on_surface,
            "on_surface_variant": l_on_surf_var,
            "outline_variant": l_outline,
            "css": {
                "--md-sys-color-surface": _rgb_hex(l_surface),
                "--md-sys-color-surface-container-low": _rgb_hex(l_surf_low),
                "--md-sys-color-surface-container": _rgb_hex(l_surf_cont),
                "--md-sys-color-surface-container-high": _rgb_hex(l_surf_high),
                "--md-sys-color-primary": _rgb_hex(l_primary),
                "--md-sys-color-on-primary": _rgb_hex(l_on_primary),
                "--md-sys-color-primary-container": _rgb_hex(l_prim_cont),
                "--md-sys-color-on-primary-container": _rgb_hex(l_on_prim_cont),
                "--md-sys-color-secondary": _rgb_hex(l_secondary),
                "--md-sys-color-secondary-container": _rgb_hex(l_sec_cont),
                "--md-sys-color-on-secondary-container": _rgb_hex(l_on_sec_cont),
                "--md-sys-color-on-surface": _rgb_hex(l_on_surface),
                "--md-sys-color-on-surface-variant": _rgb_hex(l_on_surf_var),
                "--md-sys-color-outline-variant": _rgb_hex(l_outline),
                "--md-sys-color-shadow": "#000000",
            },
        },
    }


async def load_cover_bytes(image_id: str) -> tuple[bytes | None, str]:
    """Load cover bytes via KWhale images cache, falling back gracefully on error."""
    if not image_id or image_id == images.PLACEHOLDER_ID:
        return None, "image/png"
    try:
        row = await images._load(image_id)
        data = row.get("data")
        mime = str(row.get("mime") or "image/jpeg")
        if data:
            return bytes(data), mime
    except Exception:  # noqa: BLE001 — fallback cleanly to Monet gradient artwork
        pass
    return None, "image/png"


async def get_monet_tokens(image_id: str, seed_text: str) -> dict[str, dict[str, Any]]:
    cache_key = f"{image_id}:{seed_text}"
    if cache_key in _palette_cache:
        _palette_cache.move_to_end(cache_key)
        return _palette_cache[cache_key]
    raw, _ = await load_cover_bytes(image_id)
    seed = await asyncio.to_thread(_extract_seed, raw, seed_text)
    tokens = build_monet_tokens(seed)
    _palette_cache[cache_key] = tokens
    while len(_palette_cache) > _PALETTE_CACHE_MAX:
        _palette_cache.popitem(last=False)
    return tokens


def _load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = _BOLD_FONT_CANDIDATES if bold else _REG_FONT_CANDIDATES
    for path in candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size=size)
            except OSError:
                continue
    return ImageFont.load_default()


def _text_width(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont | ImageFont.FreeTypeFont) -> float:
    bbox = draw.textbbox((0, 0), text, font=font)
    return float(bbox[2] - bbox[0])


def _wrap_lines(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.ImageFont | ImageFont.FreeTypeFont,
    max_width: int,
    max_lines: int = 2,
) -> list[str]:
    words = (text or "").split()
    if not words:
        return [""]
    lines: list[str] = []
    cur = ""
    for word in words:
        trial = f"{cur} {word}".strip() if cur else word
        if _text_width(draw, trial, font) <= max_width:
            cur = trial
        else:
            if cur:
                lines.append(cur)
                if len(lines) == max_lines:
                    break
                cur = word
            else:
                cur = word
    if len(lines) < max_lines and cur:
        lines.append(cur)
    if len(lines) == max_lines:
        consumed = sum(len(line.split()) for line in lines)
        if consumed < len(words) or _text_width(draw, lines[-1], font) > max_width:
            last = lines[-1].rstrip()
            while len(last) > 1 and _text_width(draw, last + "…", font) > max_width:
                last = last[:-1].rstrip()
            lines[-1] = (last + "…") if not last.endswith("…") else last
    return lines[:max_lines]


def _truncate_line(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.ImageFont | ImageFont.FreeTypeFont,
    max_width: int,
) -> str:
    if _text_width(draw, text, font) <= max_width:
        return text
    out = text.rstrip()
    while len(out) > 1 and _text_width(draw, out + "…", font) > max_width:
        out = out[:-1].rstrip()
    return out + "…"


def _rounded_artwork(cover_rgb: Image.Image, size: int, radius: int, circle: bool = False) -> Image.Image:
    sq = cover_rgb.resize((size, size), Image.Resampling.LANCZOS).convert("RGBA")
    scale = 2
    mask = Image.new("L", (size * scale, size * scale), 0)
    mdraw = ImageDraw.Draw(mask)
    if circle:
        mdraw.ellipse((0, 0, size * scale - 1, size * scale - 1), fill=255)
    else:
        mdraw.rounded_rectangle(
            (0, 0, size * scale - 1, size * scale - 1),
            radius=radius * scale,
            fill=255,
        )
    mask_small = mask.resize((size, size), Image.Resampling.LANCZOS)
    sq.putalpha(mask_small)
    return sq


def _monogram_artwork(
    seed_text: str,
    title: str,
    size: int,
    scheme: dict[str, Any],
) -> Image.Image:
    img = Image.new("RGB", (size, size), scheme["primary_container"])
    draw = ImageDraw.Draw(img)
    bg1 = scheme["primary_container"]
    bg2 = scheme["surface_high"]
    for y in range(size):
        t = y / max(1, size - 1)
        row_c = (
            int(bg1[0] * (1 - t) + bg2[0] * t),
            int(bg1[1] * (1 - t) + bg2[1] * t),
            int(bg1[2] * (1 - t) + bg2[2] * t),
        )
        draw.line([(0, y), (size, y)], fill=row_c)
    glyph_font = _load_font(int(size * 0.44), bold=True)
    ch = (title or seed_text or "K").strip()[:1].upper()
    bbox = draw.textbbox((0, 0), ch, font=glyph_font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text(
        ((size - tw) // 2 - bbox[0], (size - th) // 2 - bbox[1]),
        ch,
        font=glyph_font,
        fill=scheme["on_primary_container"],
    )
    return img


@lru_cache(maxsize=4)
def _load_kwhale_logo_badge(size: int) -> Image.Image | None:
    logo_path = Path(__file__).resolve().parent.parent / "static" / "share" / "kwhale-logo.png"
    if not logo_path.is_file():
        return None
    try:
        with Image.open(logo_path) as src:
            return src.convert("RGBA").resize((size, size), Image.Resampling.LANCZOS)
    except Exception:  # noqa: BLE001 — clean fallback if logo asset missing
        return None


def _render_sync(
    record: ShareRecord,
    cover_bytes: bytes | None,
    scheme: dict[str, Any],
    fmt: str,
    origin: str = "",
) -> bytes:
    d = scheme["dark"]
    surface = d["surface"]
    primary = d["primary"]
    on_primary = d["on_primary"]
    prim_cont = d["primary_container"]
    on_prim_cont = d["on_primary_container"]
    on_surface = d["on_surface"]
    on_surf_var = d["on_surface_variant"]
    outline = d["outline_variant"]

    # Base tonal canvas
    canvas = Image.new("RGB", (_CARD_W, _CARD_H), surface)
    cover_rgb: Image.Image | None = None
    if cover_bytes:
        try:
            with Image.open(io.BytesIO(cover_bytes)) as src:
                cover_rgb = src.convert("RGB")
        except Exception:  # noqa: BLE001
            cover_rgb = None

    if cover_rgb is not None:
        # Gentle Material You atmospheric wash behind artwork from blurred cover.
        bg = cover_rgb.resize((_CARD_W, _CARD_H), Image.Resampling.BILINEAR)
        bg = bg.filter(ImageFilter.GaussianBlur(radius=54))
        tint = Image.new("RGB", (_CARD_W, _CARD_H), surface)
        canvas = Image.blend(bg, tint, 0.78)
    else:
        draw_bg = ImageDraw.Draw(canvas)
        for y in range(_CARD_H):
            t = y / (_CARD_H - 1)
            mix = (
                int(surface[0] * (1 - t * 0.35) + prim_cont[0] * (t * 0.35)),
                int(surface[1] * (1 - t * 0.35) + prim_cont[1] * (t * 0.35)),
                int(surface[2] * (1 - t * 0.35) + prim_cont[2] * (t * 0.35)),
            )
            draw_bg.line([(0, y), (_CARD_W, y)], fill=mix)

    draw = ImageDraw.Draw(canvas, "RGBA")

    # Left column artwork
    art_size = 420
    art_x = 68
    art_y = (_CARD_H - art_size) // 2
    src_art = cover_rgb or _monogram_artwork(record.subtitle or record.title, record.title, art_size, d)
    is_artist = record.kind == "artist"
    art_rgba = _rounded_artwork(src_art, art_size, radius=28, circle=is_artist)

    # Soft tonal halo around artwork (clean M3 containment without fake heavy drop-shadow)
    if is_artist:
        draw.ellipse(
            (art_x - 3, art_y - 3, art_x + art_size + 3, art_y + art_size + 3),
            outline=outline + (180,),
            width=3,
        )
    else:
        draw.rounded_rectangle(
            (art_x - 2, art_y - 2, art_x + art_size + 2, art_y + art_size + 2),
            radius=30,
            outline=outline + (160,),
            width=2,
        )
    canvas.paste(art_rgba, (art_x, art_y), art_rgba)

    # Right column editorial M3 typography
    tx = art_x + art_size + 56
    max_tw = _CARD_W - tx - 64

    kind_labels = {"track": "ТРЕК · KWHALE", "album": "АЛЬБОМ · KWHALE", "artist": "ИСПОЛНИТЕЛЬ · KWHALE"}
    kind_text = kind_labels.get(record.kind, "KWHALE")
    f_badge = _load_font(17, bold=True)
    bw = int(_text_width(draw, kind_text, f_badge)) + 32
    draw.rounded_rectangle((tx, 108, tx + bw, 144), radius=18, fill=prim_cont + (255,))
    draw.text((tx + 16, 116), kind_text, font=f_badge, fill=on_prim_cont + (255,))

    # Title (up to 2 lines)
    title_size = 50 if len(record.title) <= 28 else 42
    f_title = _load_font(title_size, bold=True)
    title_lines = _wrap_lines(draw, record.title, f_title, max_tw, max_lines=2)
    cy = 172
    line_step = int(title_size * 1.18)
    for line in title_lines:
        draw.text((tx, cy), line, font=f_title, fill=on_surface + (255,))
        cy += line_step

    # Subtitle & metadata
    f_sub = _load_font(30, bold=False)
    if record.subtitle and record.subtitle != "Исполнитель":
        sub = _truncate_line(draw, record.subtitle, f_sub, max_tw)
        draw.text((tx, cy + 8), sub, font=f_sub, fill=primary + (255,))
        cy += 48

    f_meta = _load_font(23, bold=False)
    meta_text = record.meta or ("Исполнитель KWhale" if is_artist else "")
    if meta_text:
        meta_line = _truncate_line(draw, meta_text, f_meta, max_tw)
        draw.text((tx, cy + 8), meta_line, font=f_meta, fill=on_surf_var + (255,))

    # Quiet M3 footer bar
    draw.line([(tx, 468), (tx + max_tw, 468)], fill=outline + (180,), width=1)

    # KWhale orca logo + service summary
    logo_badge = _load_kwhale_logo_badge(38)
    if logo_badge is not None:
        canvas.paste(logo_badge, (tx, 492), logo_badge)
    else:
        draw.ellipse((tx, 492, tx + 38, 530), fill=primary + (255,))
        f_k = _load_font(21, bold=True)
        kb = draw.textbbox((0, 0), "K", font=f_k)
        draw.text(
            (tx + 19 - (kb[2] - kb[0]) // 2 - kb[0], 492 + 19 - (kb[3] - kb[1]) // 2 - kb[1]),
            "K",
            font=f_k,
            fill=on_primary + (255,),
        )
    f_foot = _load_font(20, bold=False)
    labels = [p.label for p in record.platforms]
    foot_max = max_tw - 54
    foot_line = (origin or "localhost:19001").removeprefix("https://").removeprefix("http://").rstrip("/")
    if labels:
        chosen: list[str] = []
        for idx, lbl in enumerate(labels):
            candidate_list = chosen + [lbl]
            rem = len(labels) - (idx + 1)
            candidate = " · ".join(candidate_list) + (f" + ещё {rem}" if rem > 0 else "")
            if _text_width(draw, candidate, f_foot) <= foot_max or not chosen:
                chosen.append(lbl)
            else:
                break
        rem_final = len(labels) - len(chosen)
        raw_foot = " · ".join(chosen) + (f" + ещё {rem_final}" if rem_final > 0 else "")
        foot_line = _truncate_line(draw, raw_foot, f_foot, foot_max)
    draw.text((tx + 52, 500), foot_line, font=f_foot, fill=on_surf_var + (255,))

    buf = io.BytesIO()
    if fmt == "jpg":
        canvas.save(buf, format="JPEG", quality=88, optimize=True, progressive=True)
    else:
        canvas.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


async def render_share_card(record: ShareRecord, fmt: str = "png", origin: str = "") -> tuple[bytes, str, str]:
    """Render (with bounded LRU cache) a 1200x630 Material You 3 OG preview card."""
    norm_fmt = "jpg" if fmt.lower() in ("jpg", "jpeg") else "png"
    ver_hash = hashlib.sha1(
        f"{record.updated_at.isoformat()}|{record.title}|{record.subtitle}|{record.image_id}|{len(record.platforms)}|{origin}".encode()
    ).hexdigest()[:12]
    cache_key = (f"{record.id}:{ver_hash}", norm_fmt)
    if cache_key in _card_cache:
        _card_cache.move_to_end(cache_key)
        return _card_cache[cache_key]

    cover_bytes, _ = await load_cover_bytes(record.image_id)
    tokens = await get_monet_tokens(record.image_id, record.subtitle or record.title)
    raw = await asyncio.to_thread(_render_sync, record, cover_bytes, tokens, norm_fmt, origin)
    mime = "image/jpeg" if norm_fmt == "jpg" else "image/png"
    etag = hashlib.sha256(raw).hexdigest()[:32]
    entry = (raw, mime, etag)
    _card_cache[cache_key] = entry
    while len(_card_cache) > _CARD_CACHE_MAX:
        _card_cache.popitem(last=False)
    return entry
