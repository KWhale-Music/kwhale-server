"""Query understanding (`QueryPlan`), mood blueprints, and deterministic fallback.

The LLM never emits SQL and never returns track IDs directly; it translates a
free-text request into a structured `QueryPlan` (target audio profile, canonical
subgenres/moods/textures/scenes, hard exclusions, semantic query for
Giga-Embeddings, taste blend, and sequencing arc).
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, Literal

import httpx
from pydantic import BaseModel, Field

from ...config import settings
from ..genres import family_set
from .tagger import normalize_tag_list

log = logging.getLogger("kwhale.rec.llm")

PRESETS: dict[str, dict] = {
    "default": {"pref": 0.65, "ctx": 0.20, "novelty": 0.15},
    "morning": {"pref": 0.55, "ctx": 0.30, "novelty": 0.15},
    "focus": {"pref": 0.50, "ctx": 0.15, "novelty": 0.35},
    "evening_chill": {"pref": 0.45, "ctx": 0.35, "novelty": 0.20},
    "party": {"pref": 0.60, "ctx": 0.25, "novelty": 0.15},
    "night": {"pref": 0.50, "ctx": 0.30, "novelty": 0.20},
}

_KEYWORDS = {
    "morning": ("утро", "утрен", "бодр", "зарядк", "morning"),
    "focus": ("работ", "фокус", "концентрац", "учеб", "код", "focus", "study"),
    "evening_chill": ("вечер", "спокой", "чилл", "расслаб", "тих", "медитац", "chill", "evening"),
    "party": ("вечеринк", "танц", "клуб", "party", "движ", "тусов"),
    "night": ("ноч", "сон", "засып", "ночн", "night"),
}

_FILTER_KEYS = {
    "energy_min": (0.0, 1.0),
    "energy_max": (0.0, 1.0),
    "valence_min": (0.0, 1.0),
    "valence_max": (0.0, 1.0),
    "bpm_min": (0.0, 300.0),
    "bpm_max": (0.0, 300.0),
    "acousticness_min": (0.0, 1.0),
    "acousticness_max": (0.0, 1.0),
    "instrumentalness_min": (0.0, 1.0),
}

_AUDIO_TARGET_RANGES = {
    "energy": (0.0, 1.0),
    "valence": (0.0, 1.0),
    "danceability": (0.0, 1.0),
    "bpm": (40.0, 220.0),
    "acousticness": (0.0, 1.0),
    "instrumentalness": (0.0, 1.0),
}

_PLAN_CACHE_TTL = 3600.0
_PLAN_CACHE_MAX = 512
_plan_cache: dict[str, tuple[float, dict[str, Any]]] = {}

_VOCAB_CACHE_TTL = 600.0
_vocab_cache: tuple[float, str] | None = None

_LABELS = {
    "genre": "genres",
    "subgenre": "subgenres",
    "mood": "moods",
    "texture": "textures",
    "scene": "scenes",
}

_VOCAB_SQL = """
SELECT 'subgenre' AS kind, u.tag AS tag, COUNT(*) AS n
FROM track_features tf, unnest(tf.subgenres) AS u(tag)
GROUP BY u.tag
UNION ALL
SELECT 'mood', u.tag, COUNT(*)
FROM track_features tf, unnest(tf.moods) AS u(tag)
GROUP BY u.tag
UNION ALL
SELECT 'texture', u.tag, COUNT(*)
FROM track_features tf, unnest(tf.textures) AS u(tag)
GROUP BY u.tag
UNION ALL
SELECT 'scene', u.tag, COUNT(*)
FROM track_features tf, unnest(tf.scenes) AS u(tag)
GROUP BY u.tag
UNION ALL
SELECT 'genre', u.tag, COUNT(DISTINCT t.id)
FROM tracks t, unnest(t.genres_norm) AS u(tag)
WHERE EXISTS (
    SELECT 1 FROM track_sources s
    WHERE s.track_id = t.id AND s.connector = 'local' AND s.available
)
GROUP BY u.tag
"""


async def library_vocabulary() -> str:
    """Snapshot of tags that actually exist in the local library.

    The LLM otherwise invents plausible but absent tags (e.g. "saxophone"
    instead of the real "jazz_samples").  Cached for ten minutes.
    """
    global _vocab_cache
    now = time.monotonic()
    if _vocab_cache is not None and now - _vocab_cache[0] < _VOCAB_CACHE_TTL:
        return _vocab_cache[1]

    try:
        from ... import db

        rows = await db.fetch_all(_VOCAB_SQL)
    except Exception:
        log.exception("library vocabulary query failed")
        return ""

    buckets: dict[str, list[tuple[int, str]]] = {key: [] for key in _LABELS}
    for row in rows:
        kind = str(row.get("kind") or "")
        tag = str(row.get("tag") or "").strip()
        if kind not in buckets or not tag:
            continue
        try:
            count = int(row.get("n") or 0)
        except (TypeError, ValueError):
            count = 0
        buckets[kind].append((count, tag))

    lines = ["What actually exists in the user's local library (tag — number of tracks):"]
    for kind, label in _LABELS.items():
        entries = sorted(buckets[kind], key=lambda item: (-item[0], item[1]))
        if entries:
            lines.append(f"{label}: " + ", ".join(f"{tag} ({n})" for n, tag in entries))
    text = "\n".join(lines) if len(lines) > 1 else ""
    if text:
        _vocab_cache = (now, text)
    return text


def clear_vocabulary_cache() -> None:
    global _vocab_cache
    _vocab_cache = None


class QueryPlan(BaseModel):
    intent: Literal[
        "personal",
        "discovery",
        "mood",
        "genre_style",
        "activity_scene",
        "similar_to",
        "hybrid",
    ] = "hybrid"
    preset: str = "default"
    semantic_query: str = ""
    include_genres: list[str] = Field(default_factory=list)
    exclude_genres: list[str] = Field(default_factory=list)
    target_subgenres: list[str] = Field(default_factory=list)
    exclude_subgenres: list[str] = Field(default_factory=list)
    target_moods: list[str] = Field(default_factory=list)
    exclude_moods: list[str] = Field(default_factory=list)
    target_textures: list[str] = Field(default_factory=list)
    exclude_textures: list[str] = Field(default_factory=list)
    target_scenes: list[str] = Field(default_factory=list)
    target_language: str | None = None
    year_min: int | None = None
    year_max: int | None = None
    reference_artists: list[str] = Field(default_factory=list)
    exclude_artists: list[str] = Field(default_factory=list)
    audio_target: dict[str, float] = Field(default_factory=dict)
    axis_weights: dict[str, float] = Field(default_factory=dict)
    hard_bounds: dict[str, float] = Field(default_factory=dict)
    taste_blend: float = 0.25
    arc: Literal["wave", "build_up", "wind_down", "steady"] = "wave"
    exploration: float = 0.15
    size: int = 30
    weights: dict[str, float] = Field(default_factory=lambda: dict(PRESETS["default"]))
    source: str = "fallback"


MOOD_PLANS: dict[str, dict[str, Any]] = {
    "morning": {
        "intent": "mood",
        "preset": "morning",
        "semantic_query": "бодрое светлое утро, свежая позитивная музыка, тёплый инди-поп, лёгкий грув, uplifting bright fresh morning",
        "target_moods": ["uplifting", "bright", "warm", "happy", "energetic"],
        "exclude_moods": ["dark", "sad", "aggressive"],
        "target_scenes": ["morning", "walk"],
        "audio_target": {"energy": 0.62, "valence": 0.68, "bpm": 114.0, "danceability": 0.58},
        "axis_weights": {"energy": 0.75, "valence": 0.85, "bpm": 0.45, "danceability": 0.50},
        "hard_bounds": {"energy_min": 0.38, "valence_min": 0.42},
        "taste_blend": 0.40,
        "arc": "build_up",
        "size": 30,
    },
    "focus": {
        "intent": "mood",
        "preset": "focus",
        "semantic_query": "фокус концентрация работа учёба спокойный ровный бит инструментал эмбиент лоу-фай неоклассика без агрессии focus study steady",
        "target_moods": ["calm", "chill", "atmospheric", "warm"],
        "exclude_moods": ["aggressive", "party", "intense"],
        "target_textures": ["lo_fi", "piano", "synth", "acoustic", "ambient_drone", "keys"],
        "target_scenes": ["focus", "study", "relax"],
        "audio_target": {"energy": 0.38, "valence": 0.48, "bpm": 96.0, "instrumentalness": 0.65},
        "axis_weights": {"energy": 0.80, "bpm": 0.40, "instrumentalness": 0.75},
        "hard_bounds": {"energy_max": 0.62},
        "taste_blend": 0.30,
        "arc": "steady",
        "size": 30,
    },
    "evening_chill": {
        "intent": "mood",
        "preset": "evening_chill",
        "semantic_query": "расслабленный тёплый вечер чилл мягкое звучание уютный джаз рнб инди соул evening chill warm dreamy relaxed",
        "target_moods": ["chill", "calm", "warm", "dreamy", "melancholic", "nostalgic"],
        "exclude_moods": ["aggressive"],
        "target_scenes": ["evening", "relax", "sunset", "rainy"],
        "audio_target": {"energy": 0.38, "valence": 0.46, "bpm": 92.0, "acousticness": 0.45},
        "axis_weights": {"energy": 0.85, "valence": 0.55, "bpm": 0.45},
        "hard_bounds": {"energy_max": 0.62},
        "taste_blend": 0.38,
        "arc": "wind_down",
        "size": 30,
    },
    "party": {
        "intent": "mood",
        "preset": "party",
        "semantic_query": "энергичная танцевальная вечеринка клуб драйв бас хайп ритм party dance club hype energetic upbeat",
        "target_moods": ["energetic", "party", "euphoric", "happy", "intense"],
        "exclude_moods": ["sad", "calm", "sleep"],
        "target_scenes": ["party", "club", "workout"],
        "audio_target": {"energy": 0.82, "valence": 0.66, "bpm": 124.0, "danceability": 0.72},
        "axis_weights": {"energy": 0.90, "danceability": 0.85, "bpm": 0.60, "valence": 0.45},
        "hard_bounds": {"energy_min": 0.56},
        "taste_blend": 0.38,
        "arc": "build_up",
        "size": 30,
    },
    "night": {
        "intent": "mood",
        "preset": "night",
        "semantic_query": "глубокая ночь ночная поездка тёмная атмосферная музыка меланхоличный синт тёмный бит late night dark nocturnal atmospheric drive",
        "target_moods": ["nocturnal", "dark", "atmospheric", "melancholic", "dreamy"],
        "exclude_moods": ["party", "happy"],
        "target_scenes": ["late_night", "night_drive", "rainy"],
        "audio_target": {"energy": 0.44, "valence": 0.34, "bpm": 98.0},
        "axis_weights": {"energy": 0.70, "valence": 0.80, "bpm": 0.40},
        "hard_bounds": {"valence_max": 0.60, "energy_max": 0.72},
        "taste_blend": 0.38,
        "arc": "steady",
        "size": 30,
    },
}


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(value)))


def preset_for_prompt(prompt: str) -> str:
    text = (prompt or "").lower()
    for preset, words in _KEYWORDS.items():
        if any(word in text for word in words):
            return preset
    return "default"


def validate_plan(raw: dict[str, Any] | None, *, fallback_prompt: str = "") -> QueryPlan:
    raw = raw or {}
    preset = raw.get("preset")
    if preset not in PRESETS:
        preset = preset_for_prompt(fallback_prompt) if fallback_prompt else "default"

    weights = dict(PRESETS[preset])
    incoming_w = raw.get("weights") or {}
    for key in ("pref", "ctx", "novelty"):
        if key in incoming_w:
            try:
                weights[key] = _clamp(float(incoming_w[key]), 0.0, 1.0)
            except (TypeError, ValueError):
                continue
    total: float = sum(weights.values())
    if total <= 0:
        weights = dict(PRESETS["default"])
        total = 1.0
    weights = {key: value / total for key, value in weights.items()}

    # Support legacy `filters` dict alongside new `hard_bounds` / `include_genres`
    legacy_filters = raw.get("filters") or {}
    hard_bounds: dict[str, float] = {}
    for src in (legacy_filters, raw.get("hard_bounds") or {}):
        if not isinstance(src, dict):
            continue
        for key, (lo, hi) in _FILTER_KEYS.items():
            if key in src and src[key] is not None:
                try:
                    hard_bounds[key] = _clamp(float(src[key]), lo, hi)
                except (TypeError, ValueError):
                    continue

    include_genres = sorted(
        family_set(
            [str(x) for x in (raw.get("include_genres") or legacy_filters.get("genres_include") or [])]
        )
    )[:10]
    exclude_genres = sorted(
        family_set(
            [str(x) for x in (raw.get("exclude_genres") or legacy_filters.get("genres_exclude") or [])]
        )
    )[:10]
    # An explicit exclude overrides accidental include
    include_genres = [g for g in include_genres if g not in exclude_genres]

    exclude_artists_raw = raw.get("exclude_artists") or legacy_filters.get("exclude_artists") or []
    exclude_artists = [str(a).strip() for a in exclude_artists_raw if str(a).strip()][:20]
    reference_artists = [
        str(a).strip() for a in (raw.get("reference_artists") or []) if str(a).strip()
    ][:10]

    audio_target: dict[str, float] = {}
    for key, (lo, hi) in _AUDIO_TARGET_RANGES.items():
        val = (raw.get("audio_target") or {}).get(key)
        if val is not None:
            try:
                audio_target[key] = _clamp(float(val), lo, hi)
            except (TypeError, ValueError):
                continue

    axis_weights: dict[str, float] = {}
    for key in _AUDIO_TARGET_RANGES:
        val = (raw.get("axis_weights") or {}).get(key)
        if val is not None:
            try:
                axis_weights[key] = _clamp(float(val), 0.0, 1.0)
            except (TypeError, ValueError):
                continue

    lang = str(raw.get("target_language") or "").strip().lower()
    if lang not in ("ru", "en", "instrumental"):
        lang = ""

    def _year(val: Any) -> int | None:
        if val is None:
            return None
        try:
            y = int(val)
            return y if 1920 <= y <= 2035 else None
        except (TypeError, ValueError):
            return None

    year_min = _year(raw.get("year_min"))
    year_max = _year(raw.get("year_max"))
    if year_min and year_max and year_min > year_max:
        year_min, year_max = year_max, year_min

    try:
        taste_blend = _clamp(float(raw.get("taste_blend", 0.25)), 0.0, 0.85)
    except (TypeError, ValueError):
        taste_blend = 0.25

    try:
        exploration = _clamp(float(raw.get("exploration", 0.18)), 0.0, 0.35)
    except (TypeError, ValueError):
        exploration = 0.18

    try:
        size = int(_clamp(float(raw.get("size", 30)), 10, 100))
    except (TypeError, ValueError):
        size = 30

    _ARCS: dict[str, Literal["wave", "build_up", "wind_down", "steady"]] = {
        "wave": "wave",
        "build_up": "build_up",
        "wind_down": "wind_down",
        "steady": "steady",
    }
    arc = _ARCS.get(str(raw.get("arc") or ""), "wave")

    _INTENTS: dict[str, Literal["personal", "discovery", "mood", "genre_style", "activity_scene", "similar_to", "hybrid"]] = {
        "personal": "personal",
        "discovery": "discovery",
        "mood": "mood",
        "genre_style": "genre_style",
        "activity_scene": "activity_scene",
        "similar_to": "similar_to",
        "hybrid": "hybrid",
    }
    intent = _INTENTS.get(str(raw.get("intent") or ""), "hybrid")

    sem_query = str(raw.get("semantic_query") or fallback_prompt or "").strip()[:500]

    return QueryPlan(
        intent=intent,
        preset=preset,
        semantic_query=sem_query,
        include_genres=include_genres,
        exclude_genres=exclude_genres,
        target_subgenres=normalize_tag_list(raw.get("target_subgenres"), limit=10),
        exclude_subgenres=normalize_tag_list(raw.get("exclude_subgenres"), limit=10),
        target_moods=normalize_tag_list(raw.get("target_moods"), limit=10),
        exclude_moods=normalize_tag_list(raw.get("exclude_moods"), limit=10),
        target_textures=normalize_tag_list(raw.get("target_textures"), limit=10),
        exclude_textures=normalize_tag_list(raw.get("exclude_textures"), limit=10),
        target_scenes=normalize_tag_list(raw.get("target_scenes"), limit=10),
        target_language=lang or None,
        year_min=year_min,
        year_max=year_max,
        reference_artists=reference_artists,
        exclude_artists=exclude_artists,
        audio_target=audio_target,
        axis_weights=axis_weights,
        hard_bounds=hard_bounds,
        taste_blend=taste_blend,
        arc=arc,
        exploration=exploration,
        size=size,
        weights=weights,
        source=str(raw.get("source") or "fallback"),
    )


def plan_for_mood(preset: str) -> QueryPlan:
    spec = MOOD_PLANS.get(preset) or MOOD_PLANS["evening_chill"]
    return validate_plan({**spec, "source": "preset"})


def plan_to_legacy_patch(plan: QueryPlan) -> dict[str, Any]:
    """Convert `QueryPlan` into legacy `filters`/`weights` shape for callers & tests."""
    filters: dict[str, Any] = dict(plan.hard_bounds)
    if plan.include_genres:
        filters["genres_include"] = list(plan.include_genres)
    if plan.exclude_genres:
        filters["genres_exclude"] = list(plan.exclude_genres)
    if plan.exclude_artists:
        filters["exclude_artists"] = list(plan.exclude_artists)
    return {
        "preset": plan.preset,
        "weights": dict(plan.weights),
        "filters": filters,
        "exploration": plan.exploration,
        "size": plan.size,
        "source": plan.source,
    }


def validate_patch(raw: dict | None) -> dict:
    """Validate legacy scoring patch (for /scoring/profiles/* and existing tests)."""
    raw = raw or {}
    preset = raw.get("preset")
    if preset not in PRESETS:
        preset = "default"

    weights = dict(PRESETS[preset])
    incoming = raw.get("weights") or {}
    for key in ("pref", "ctx", "novelty"):
        if key in incoming:
            try:
                weights[key] = _clamp(float(incoming[key]), 0.0, 1.0)
            except (TypeError, ValueError):
                continue
    total: float = sum(weights.values())
    if total <= 0:
        weights = dict(PRESETS["default"])
        total = 1.0
    weights = {key: value / total for key, value in weights.items()}

    filters: dict[str, float | list[str]] = {}
    for key, value in (raw.get("filters") or {}).items():
        if key in _FILTER_KEYS:
            lo, hi = _FILTER_KEYS[key]
            try:
                filters[key] = _clamp(float(value), lo, hi)
            except (TypeError, ValueError):
                continue
        elif key in ("genres_include", "genres_exclude"):
            if isinstance(value, list):
                filters[key] = sorted(family_set([str(item) for item in value]))[:20]
        elif key == "exclude_artists":
            if isinstance(value, list):
                filters[key] = [str(item) for item in value][:20]

    exploration = 0.2
    if "exploration" in raw:
        try:
            exploration = _clamp(float(raw["exploration"]), 0.0, 0.3)
        except (TypeError, ValueError):
            pass

    size = 40
    if "size" in raw:
        try:
            size = int(_clamp(float(raw["size"]), 10, 100))
        except (TypeError, ValueError):
            pass

    return {
        "preset": preset,
        "weights": weights,
        "filters": filters,
        "exploration": exploration,
        "size": size,
    }


def _extract_exclusions(text: str) -> tuple[str, list[str], list[str], list[str], list[str]]:
    """Parse all «без <X>» clauses into excluded genres, subgenres, moods, and artists."""
    exc_genres: set[str] = set()
    exc_subgenres: list[str] = []
    exc_moods: list[str] = []
    exc_artists: list[str] = []

    chunks: list[str] = []

    def _capture(match: re.Match[str]) -> str:
        chunks.append(match.group(1).strip())
        return " "

    cleaned = re.sub(
        r"(?:^|[,;\s])без\s+(.+?)(?=(?:[,.;!?\n]|\s+(?:на|для|под|по|из|как|в\s+духе|в\s+стиле|похоже|\d{4}\b)|$))",
        _capture,
        text,
        flags=re.IGNORECASE,
    )
    for chunk in chunks:
        for subchunk in re.split(r"\s+и\s+", chunk):
            low = subchunk.lower().strip()
            if not low:
                continue
            fams = family_set([low])
            if fams:
                exc_genres.update(fams)
                if "rap" in fams:
                    exc_subgenres.extend(["rap", "russian_rap", "trap", "drill", "phonk"])
                continue
            if any(w in low for w in ("агресс", "жест", "жёст")):
                exc_moods.extend(["aggressive", "intense"])
            elif any(w in low for w in ("груст", "меланх", "печал")):
                exc_moods.extend(["sad", "melancholic"])
            elif any(w in low for w in ("слов", "вокал", "текст")):
                exc_moods.append("vocal_heavy")
            else:
                exc_artists.append(low)

    return cleaned, sorted(exc_genres), exc_subgenres, exc_moods, exc_artists


def plan_from_prompt_fallback(prompt: str) -> QueryPlan:
    """Rich deterministic QueryPlan builder used when LLM is unconfigured or fails."""
    raw_text = (prompt or "").strip()
    text = raw_text.lower()
    preset = preset_for_prompt(text)

    positive_text, exc_genres, exc_subgenres, exc_moods, exc_artists = _extract_exclusions(text)
    if re.search(r"без\s+(?:слов|вокал)", text):
        target_lang: str | None = "instrumental"
    elif any(w in positive_text for w in ("русск", "russian", "русс", "ru ")):
        target_lang = "ru"
    elif any(w in positive_text for w in ("англ", "english", "зарубеж", "иностран")):
        target_lang = "en"
    elif any(w in positive_text for w in ("инструментал", "instrumental", "без вокала")):
        target_lang = "instrumental"
    else:
        target_lang = None

    inc_genres: set[str] = set()
    subgenres: list[str] = []
    moods: list[str] = []
    textures: list[str] = []
    scenes: list[str] = []
    audio_target: dict[str, float] = {}
    axis_weights: dict[str, float] = {}
    hard_bounds: dict[str, float] = {}
    ref_artists: list[str] = []
    arc = "wave"

    # Explicit reference artist clauses ("как X", "в стиле X", "похоже на X")
    for m in re.finditer(r"(?:как|похоже на|в духе)\s+([a-zа-яё0-9 .\-&]{2,35})", positive_text):
        cand = m.group(1).strip()
        if cand and not family_set([cand]):
            ref_artists.append(cand)

    # Year detection (e.g. "2019")
    year_min: int | None = None
    year_max: int | None = None
    ym = re.search(r"\b(19[6-9]\d|20[0-2]\d)\b", positive_text)
    if ym:
        y = int(ym.group(1))
        year_min, year_max = y - 2, y + 2

    # Special compound style patterns (textures vs pure genre OR)
    has_rap = any(w in positive_text for w in ("рэп", "rap", "хип-хоп", "хип хоп", "hip-hop", "hiphop"))
    has_jazz = any(w in positive_text for w in ("джаз", "jazz"))
    has_acoustic = any(w in positive_text for w in ("акуст", "acoustic", "гитар", "guitar", "unplugged"))
    has_phonk = any(w in positive_text for w in ("фонк", "phonk"))

    if has_rap and has_jazz and any(w in positive_text for w in ("сэмпл", "sample", "инструмент")):
        inc_genres.add("rap")
        subgenres.extend(["jazz_rap", "boom_bap", "chillhop"])
        textures.extend(["jazz_samples", "saxophone", "piano", "vinyl"])
        audio_target["acousticness"] = 0.50
        axis_weights["acousticness"] = 0.5
    elif has_rap and has_acoustic:
        inc_genres.add("rap")
        subgenres.extend(["acoustic_rap", "melodic_rap"])
        textures.extend(["acoustic_guitar", "acoustic"])
        audio_target.update({"acousticness": 0.78, "energy": 0.44})
        axis_weights.update({"acousticness": 0.85, "energy": 0.60})
        hard_bounds["energy_max"] = 0.72
    elif has_phonk:
        inc_genres.add("rap")
        subgenres.extend(["phonk", "dark_phonk"])
        textures.extend(["heavy_bass", "808_bass", "distorted", "lo_fi"])
        moods.extend(["dark", "nocturnal", "intense"])
        audio_target.update({"energy": 0.78, "valence": 0.28, "bpm": 130.0})
        axis_weights.update({"energy": 0.75, "valence": 0.60, "bpm": 0.45})
    else:
        explicit_genres = re.findall(r"(?:жанр[а-я]*|в стиле|genre)\s*[:\-]?\s*([a-zа-я\- ]{2,30})", positive_text)
        if explicit_genres:
            inc_genres.update(family_set(explicit_genres))
        else:
            inc_genres.update(family_set([positive_text]))
        if has_acoustic:
            textures.extend(["acoustic_guitar", "acoustic"])
            moods.extend(["warm", "tender"])
            audio_target["acousticness"] = 0.84
            axis_weights["acousticness"] = 0.85
            # Do not collapse acoustic requests into hard rap
            if not has_rap and "rap" not in inc_genres:
                exc_genres.append("rap") if "rap" not in exc_genres else None
        if has_jazz:
            if "jazz" not in inc_genres:
                inc_genres.add("jazz")
            # Style is expressed by tags, not just by the coarse genre family:
            # this is what finds jazz-rap / jazz-pop records in the library.
            for tag in ("jazz", "nu_jazz", "jazz_rap"):
                if tag not in subgenres:
                    subgenres.append(tag)
            for tag in ("jazz_samples", "piano", "saxophone"):
                if tag not in textures:
                    textures.append(tag)

    # Moods & scenes & energy/valence envelopes
    if any(w in positive_text for w in ("жёстк", "жестк", "агрессив", "тяжел", "тяжёл", "злой", "разрыв", "hard", "aggressive")):
        moods.extend(["aggressive", "intense", "dark"])
        audio_target.update({"energy": 0.85, "valence": 0.30})
        axis_weights.update({"energy": 0.90, "valence": 0.50})
        hard_bounds["energy_min"] = 0.55
        arc = "build_up"
    if any(w in positive_text for w in ("спокой", "тих", "чилл", "медитац", "сон", "нежн", "мягк", "calm", "chill", "tender")):
        moods.extend(["calm", "chill", "tender", "warm"])
        audio_target.update({"energy": 0.34, "bpm": 90.0})
        axis_weights.update({"energy": 0.90, "bpm": 0.50})
        hard_bounds["energy_max"] = 0.60
        hard_bounds["bpm_max"] = 122.0
        arc = "wind_down"
    if any(w in positive_text for w in ("энерг", "бодр", "танц", "трениров", "бег", "зал", "party", "workout")):
        moods.extend(["energetic", "uplifting"])
        scenes.extend(["workout", "party"])
        audio_target.update({"energy": 0.82, "danceability": 0.70, "bpm": 128.0})
        axis_weights.update({"energy": 0.90, "danceability": 0.65, "bpm": 0.55})
        hard_bounds["energy_min"] = 0.60
        arc = "build_up"
    if any(w in positive_text for w in ("груст", "меланхол", "печал", "sad", "melanch")):
        moods.extend(["melancholic", "sad"])
        audio_target["valence"] = 0.25
        axis_weights["valence"] = 0.80
        hard_bounds["valence_max"] = 0.50
    if any(w in positive_text for w in ("тёмн", "темн", "мрач", "dark")):
        moods.extend(["dark", "nocturnal"])
        audio_target["valence"] = min(audio_target.get("valence", 0.30), 0.30)
        axis_weights["valence"] = max(axis_weights.get("valence", 0.0), 0.70)
        hard_bounds["valence_max"] = min(hard_bounds.get("valence_max", 0.55), 0.55)
    if any(w in positive_text for w in ("весел", "веёл", "радост", "позитив", "happy")):
        moods.extend(["happy", "uplifting", "bright"])
        audio_target["valence"] = 0.78
        axis_weights["valence"] = 0.85
        hard_bounds["valence_min"] = 0.52
    if any(w in positive_text for w in ("дожд", "rain")):
        scenes.append("rainy")
        moods.extend(["atmospheric", "melancholic", "warm"])
    if any(w in positive_text for w in ("закат", "sunset")):
        scenes.append("sunset")
        moods.extend(["dreamy", "warm", "nostalgic"])
    if any(w in positive_text for w in ("трасс", "дорог", "машин", "ехать", "drive")):
        scenes.extend(["night_drive", "road_trip"])
        moods.append("atmospheric")
        audio_target.setdefault("energy", 0.48)
        audio_target.setdefault("bpm", 102.0)
        axis_weights.setdefault("energy", 0.75)
        axis_weights.setdefault("bpm", 0.45)
    if any(w in positive_text for w in ("девушк", "романт", "любов", "romantic")):
        scenes.append("romantic")
        moods.extend(["romantic", "sensual", "warm"])
    if target_lang == "instrumental":
        audio_target["instrumentalness"] = 0.88
        axis_weights["instrumentalness"] = 0.85
        hard_bounds["instrumentalness_min"] = 0.55

    semantic_bits = [raw_text]
    if subgenres:
        semantic_bits.append("subgenres: " + ", ".join(dict.fromkeys(subgenres)))
    if moods:
        semantic_bits.append("moods: " + ", ".join(dict.fromkeys(moods)))
    if textures:
        semantic_bits.append("textures: " + ", ".join(dict.fromkeys(textures)))
    if scenes:
        semantic_bits.append("scenes: " + ", ".join(dict.fromkeys(scenes)))

    has_specific = bool(inc_genres or subgenres or moods or textures or scenes or ref_artists)
    taste_blend = 0.22 if has_specific else 0.45

    return validate_plan(
        {
            "intent": "genre_style" if inc_genres or subgenres else ("activity_scene" if scenes else "mood"),
            "preset": preset,
            "semantic_query": " | ".join(semantic_bits),
            "include_genres": sorted(inc_genres),
            "exclude_genres": exc_genres,
            "target_subgenres": subgenres,
            "exclude_subgenres": exc_subgenres,
            "target_moods": moods,
            "exclude_moods": exc_moods,
            "target_textures": textures,
            "target_scenes": scenes,
            "target_language": target_lang,
            "year_min": year_min,
            "year_max": year_max,
            "reference_artists": ref_artists,
            "exclude_artists": exc_artists,
            "audio_target": audio_target,
            "axis_weights": axis_weights,
            "hard_bounds": hard_bounds,
            "taste_blend": taste_blend,
            "arc": arc,
            "source": "fallback",
        },
        fallback_prompt=raw_text,
    )


def patch_from_prompt(prompt: str) -> dict:
    """Backward-compatible deterministic patch helper."""
    return plan_to_legacy_patch(plan_from_prompt_fallback(prompt))


def merge_profile(base: dict, patch: dict) -> dict:
    merged = dict(base)
    merged["pref"] = patch["weights"]["pref"]
    merged["ctx"] = patch["weights"]["ctx"]
    merged["novelty"] = patch["weights"]["novelty"]
    merged["exploration"] = patch["exploration"]
    merged["filters"] = patch.get("filters") or {}
    return merged


def to_profile(patch: dict) -> dict:
    """Flatten a validated patch into the profile shape stored in the database."""
    return {
        "preset": patch.get("preset", "default"),
        "pref": patch["weights"]["pref"],
        "ctx": patch["weights"]["ctx"],
        "novelty": patch["weights"]["novelty"],
        "exploration": patch["exploration"],
        "filters": patch.get("filters") or {},
    }


_PLAN_SYSTEM_PROMPT = """You are a music recommendation architect and playlist query parser.
The user writes a free-text request (often in Russian). Translate it into a strict JSON QueryPlan.

Return STRICT JSON only, no prose:
{
  "intent": "personal" | "discovery" | "mood" | "genre_style" | "activity_scene" | "similar_to" | "hybrid",
  "preset": "default" | "morning" | "focus" | "evening_chill" | "party" | "night",
  "semantic_query": "<rich bilingual RU+EN description of mood, instrumentation, subgenre, scene, vocals for semantic vector retrieval>",
  "include_genres": ["<canonical family>"],
  "exclude_genres": ["<canonical family>"],
  "target_subgenres": ["<snake_case subgenre>"],
  "exclude_subgenres": ["<snake_case subgenre>"],
  "target_moods": ["<snake_case mood>"],
  "exclude_moods": ["<snake_case mood>"],
  "target_textures": ["<snake_case instrument/texture>"],
  "exclude_textures": ["<snake_case texture>"],
  "target_scenes": ["<snake_case scene/activity>"],
  "target_language": "ru" | "en" | "instrumental" | null,
  "year_min": <int or null>,
  "year_max": <int or null>,
  "reference_artists": ["<artist name if mentioned>"],
  "exclude_artists": ["<artist name if excluded>"],
  "audio_target": {
    "energy": <0..1>, "valence": <0..1>, "bpm": <40..220>,
    "danceability": <0..1>, "acousticness": <0..1>, "instrumentalness": <0..1>
  },
  "axis_weights": {"energy": <0..1>, "valence": <0..1>, "bpm": <0..1>, "danceability": <0..1>, "acousticness": <0..1>, "instrumentalness": <0..1>},
  "hard_bounds": {"energy_min": <0..1>, "energy_max": <0..1>, "valence_min": <0..1>, "valence_max": <0..1>, "bpm_min": <0..300>, "bpm_max": <0..300>, "acousticness_min": <0..1>, "instrumentalness_min": <0..1>},
  "taste_blend": <0.0..0.8>,
  "arc": "wave" | "build_up" | "wind_down" | "steady",
  "exploration": <0..0.35>,
  "size": <10..100>
}

Critical rules:
1. Canonical families for include_genres / exclude_genres ONLY from:
   rap, pop, rock, electronic, jazz, classical, folk, rnb, reggae, soundtrack, experimental, instrumental.
2. NEVER abuse include_genres to express instrumentation/texture!
   - «акустико-гитарный рэп» -> include_genres=["rap"], target_subgenres=["acoustic_rap", "melodic_rap"], target_textures=["acoustic_guitar", "acoustic"], audio_target={"acousticness": 0.8, "energy": 0.45}. Do NOT put "folk" in include_genres!
   - «рэп с джазовыми сэмплами» -> include_genres=["rap"], target_subgenres=["jazz_rap", "boom_bap"], target_textures=["jazz_samples", "saxophone", "piano"]. Do NOT put "jazz" in include_genres!
   - «нежная акустика под гитару» -> include_genres=[], exclude_genres=["rap"], target_textures=["acoustic_guitar", "acoustic"], target_moods=["tender", "calm", "warm"], audio_target={"acousticness": 0.85, "energy": 0.32}, hard_bounds={"energy_max": 0.6}.
   - «тёмный фонк» -> include_genres=["rap"], target_subgenres=["phonk", "dark_phonk"], target_moods=["dark", "nocturnal", "intense"], target_textures=["heavy_bass", "808_bass", "distorted"].
3. Explicit exclusions («без X»): ALWAYS put into exclude_genres / exclude_subgenres / exclude_artists / exclude_moods.
4. Use lower taste_blend (0.15..0.25) for specific genre/vibe/scene prompts; higher (0.40..0.60) only for personal/open prompts.
5. Canonical tags examples:
   moods: dark, aggressive, intense, melancholic, sad, nostalgic, warm, tender, dreamy, atmospheric, calm, chill, uplifting, happy, euphoric, party, romantic, nocturnal.
   textures: acoustic_guitar, electric_guitar, piano, keys, jazz_samples, saxophone, brass, strings, 808_bass, heavy_bass, synth, live_drums, lo_fi, distorted, clean_vocals, acoustic.
   scenes: night_drive, rainy, sunset, road_trip, romantic, workout, running, focus, late_night, morning, party, club, relax, evening.
6. Keep hard_bounds minimal (ONLY strong explicit user constraints like energy_max for calm/sleep or energy_min for hard/workout). Do NOT add instrumentalness_min, acousticness_min, or bpm_min/max to hard_bounds unless the user explicitly demands instrumental ("без слов") or a strict tempo; put preferences into audio_target and axis_weights instead! Leave target_language=null unless the user explicitly asks for Russian, English, or instrumental.
7. The next system message lists tags that ACTUALLY exist in the user's library. Prefer those exact snake_case tags for target_* / exclude_*. If a concept is missing from the list, put the closest existing tag in the structured fields and keep the free-form wording in semantic_query. include_genres is a SOFT preference (a ranking boost), not a hard filter: it is fine to name a scarce genre, and a genre-adjacent track may still be a good answer. For «джаз на вечер» use include_genres=["jazz"], target_subgenres=["jazz","nu_jazz","jazz_rap"] and target_textures=["jazz_samples","piano","saxophone"] — do NOT limit the request to pure jazz.
8. «Без рэпа» excludes the dominant rap style, but jazz-rap / pop-rap records are kept by the engine (the user listens to them). Do not add extra exclusions the user did not ask for; do not turn a style word into exclude_genres."""


async def plan_from_llm(
    prompt: str,
    api_key: str | None = None,
    endpoint: str | None = None,
    model: str | None = None,
    vocabulary: str | None = None,
) -> QueryPlan | None:
    key = (api_key or "").strip() or settings.deepseek_api_key
    base = (endpoint or "").strip() or settings.deepseek_base_url
    mdl = (model or "").strip() or settings.deepseek_model
    if not key:
        return None

    messages = [{"role": "system", "content": _PLAN_SYSTEM_PROMPT}]
    if vocabulary:
        messages.append(
            {
                "role": "system",
                "content": (
                    vocabulary
                    + "\nUse only these exact tags whenever possible; for anything absent, "
                    "keep the wording in semantic_query and choose the closest existing tag."
                ),
            }
        )
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": mdl,
        "messages": messages,
        "temperature": 0.2,
        "max_tokens": 1200,
        "response_format": {"type": "json_object"},
    }
    async with httpx.AsyncClient(timeout=settings.deepseek_timeout) as client:
        response = await client.post(
            f"{base.rstrip('/')}/chat/completions",
            headers={"Authorization": f"Bearer {key}"},
            json=payload,
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]

    raw = json.loads(content)
    raw["source"] = "deepseek"
    return validate_plan(raw, fallback_prompt=prompt)


async def plan_for(
    prompt: str,
    api_key: str | None = None,
    endpoint: str | None = None,
    model: str | None = None,
) -> QueryPlan:
    """DeepSeek QueryPlan with TTL cache and rich deterministic fallback."""
    key = (api_key or "").strip() or settings.deepseek_api_key
    base = (endpoint or "").strip() or settings.deepseek_base_url
    mdl = (model or "").strip() or settings.deepseek_model
    cache_key = f"{prompt.strip().lower()}::{base}::{mdl}::{bool(key)}"
    now = time.monotonic()
    cached = _plan_cache.get(cache_key)
    if cached and now - cached[0] < _PLAN_CACHE_TTL:
        return QueryPlan(**cached[1])

    plan: QueryPlan | None = None
    if key:
        try:
            vocabulary = await library_vocabulary()
            plan = await plan_from_llm(
                prompt,
                api_key=api_key,
                endpoint=endpoint,
                model=model,
                vocabulary=vocabulary,
            )
        except Exception:
            log.exception("deepseek query plan failed, falling back to deterministic parser")

    if plan is None:
        plan = plan_from_prompt_fallback(prompt)

    if len(_plan_cache) >= _PLAN_CACHE_MAX:
        oldest = min(_plan_cache.items(), key=lambda kv: kv[1][0])[0]
        _plan_cache.pop(oldest, None)
    _plan_cache[cache_key] = (now, plan.model_dump())
    return plan


async def patch_from_llm(
    prompt: str,
    api_key: str | None = None,
    endpoint: str | None = None,
    model: str | None = None,
) -> dict | None:
    vocabulary = await library_vocabulary()
    plan = await plan_from_llm(
        prompt,
        api_key=api_key,
        endpoint=endpoint,
        model=model,
        vocabulary=vocabulary,
    )
    return plan_to_legacy_patch(plan) if plan is not None else None


async def patch_for(
    prompt: str,
    api_key: str | None = None,
    endpoint: str | None = None,
    model: str | None = None,
) -> dict:
    plan = await plan_for(prompt, api_key=api_key, endpoint=endpoint, model=model)
    return plan_to_legacy_patch(plan)
