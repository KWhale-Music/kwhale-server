"""Semantic track tagging (LLM batch enrichment + deterministic DSP/metadata fallback).

Runs offline in `indexer.py` (`kind='tags'`) for local tracks only, producing
canonical subgenres, moods, textures/instruments, scenes, language, era,
calibrated semantic energy/valence/acousticness priors, and a bilingual
vibe_summary used both by SQL GIN filters and Giga-Embeddings (`kind='text'`).
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

import httpx

from ...config import settings
from ..genres import genre_families

log = logging.getLogger("kwhale.rec.tagger")

TAGS_VERSION = "tags-v1"

TAG_ALIASES: dict[str, str] = {
    # moods
    "темный": "dark",
    "тёмный": "dark",
    "мрачный": "dark",
    "gloomy": "dark",
    "агрессивный": "aggressive",
    "жесткий": "aggressive",
    "жёсткий": "aggressive",
    "hard": "aggressive",
    "грустный": "sad",
    "печальный": "sad",
    "меланхоличный": "melancholic",
    "ностальгический": "nostalgic",
    "теплый": "warm",
    "тёплый": "warm",
    "нежный": "tender",
    "мягкий": "tender",
    "dream": "dreamy",
    "мечтательный": "dreamy",
    "атмосферный": "atmospheric",
    "спокойный": "calm",
    "тихий": "calm",
    "расслабленный": "chill",
    "чилл": "chill",
    "веселый": "happy",
    "весёлый": "happy",
    "радостный": "happy",
    "позитивный": "uplifting",
    "бодрый": "uplifting",
    "энергичный": "energetic",
    "драйвовый": "energetic",
    "романтичный": "romantic",
    "ночной": "nocturnal",
    # textures / instruments
    "акустическая гитара": "acoustic_guitar",
    "акустическая_гитара": "acoustic_guitar",
    "гитара": "acoustic_guitar",
    "guitar": "acoustic_guitar",
    "акустика": "acoustic",
    "unplugged": "acoustic",
    "электрогитара": "electric_guitar",
    "пианино": "piano",
    "рояль": "piano",
    "фортепиано": "piano",
    "клавишные": "keys",
    "джазовые сэмплы": "jazz_samples",
    "джазовый сэмпл": "jazz_samples",
    "jazz sample": "jazz_samples",
    "jazz_sample": "jazz_samples",
    "саксофон": "saxophone",
    "духовые": "brass",
    "труба": "brass",
    "струнные": "strings",
    "скрипка": "strings",
    "808": "808_bass",
    "бас": "heavy_bass",
    "тяжелый бас": "heavy_bass",
    "тяжёлый бас": "heavy_bass",
    "синт": "synth",
    "синтезатор": "synth",
    "лоу-фай": "lo_fi",
    "lofi": "lo_fi",
    "lo-fi": "lo_fi",
    "перегруз": "distorted",
    "дисторшн": "distorted",
    "чистый вокал": "clean_vocals",
    "автотюн": "autotune",
    "живые барабаны": "live_drums",
    # scenes / activities
    "ночная поездка": "night_drive",
    "поездка": "road_trip",
    "трасса": "road_trip",
    "дорога": "road_trip",
    "drive": "night_drive",
    "дождь": "rainy",
    "rain": "rainy",
    "закат": "sunset",
    "вечер": "evening",
    "поздняя ночь": "late_night",
    "утро": "morning",
    "тренировка": "workout",
    "спорт": "workout",
    "бег": "running",
    "фокус": "focus",
    "учеба": "focus",
    "работа": "focus",
    "вечеринка": "party",
    "клуб": "club",
    "отдых": "relax",
    # subgenres
    "русский рэп": "russian_rap",
    "русский_рэп": "russian_rap",
    "rusrap": "russian_rap",
    "фонк": "phonk",
    "темный фонк": "dark_phonk",
    "тёмный фонк": "dark_phonk",
    "трэп": "trap",
    "бум-бэп": "boom_bap",
    "бум бэп": "boom_bap",
    "джаз рэп": "jazz_rap",
    "джаз-рэп": "jazz_rap",
    "акустический рэп": "acoustic_rap",
    "инди рок": "indie_rock",
    "инди-рок": "indie_rock",
    "пост-панк": "post_punk",
    "синтвейв": "synthwave",
    "эмбиент": "ambient",
    "шансон": "chanson",
    "фолк": "folk",
    "джаз": "jazz",
}


def normalize_tag(raw: str) -> str:
    text = re.sub(r"[\s\-]+", " ", (raw or "").strip().lower()).strip()
    if not text:
        return ""
    if text in TAG_ALIASES:
        return TAG_ALIASES[text]
    slug = re.sub(r"[^a-zа-яё0-9]+", "_", text).strip("_")
    return TAG_ALIASES.get(slug, slug)


def normalize_tag_list(items: Any, limit: int = 10) -> list[str]:
    if not isinstance(items, (list, tuple, set)):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for raw in items:
        norm = normalize_tag(str(raw))
        if norm and norm not in seen:
            seen.add(norm)
            out.append(norm)
            if len(out) >= limit:
                break
    return out


def _detect_language(title: str, artists: list[str], lyrics: str | None) -> str | None:
    sample = f"{title} {' '.join(artists)} {(lyrics or '')[:400]}"
    cyr = sum(1 for ch in sample if "а" <= ch.lower() <= "я" or ch.lower() == "ё")
    lat = sum(1 for ch in sample if "a" <= ch.lower() <= "z")
    total = cyr + lat
    if total == 0:
        return None
    if cyr >= max(2, int(total * 0.2)):
        return "ru"
    if lat >= 3:
        return "en"
    return None


def _era_from_year(year: int | None) -> str | None:
    if not year or year < 1920 or year > 2035:
        return None
    decade = (int(year) // 10) * 10
    return f"{decade}s"


def _parse_extra(row: dict[str, Any]) -> dict[str, Any]:
    raw = row.get("dsp_extra")
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {}
        except ValueError:
            return {}
    return {}


def heuristic_tags(row: dict[str, Any]) -> dict[str, Any]:
    """Fast deterministic tags from metadata + Essentia DSP (always works offline)."""
    title = str(row.get("title") or "")
    artists = [str(a) for a in (row.get("artists") or []) if a]
    raw_genre = str(row.get("genre") or "").strip()
    families = list(row.get("genres_norm") or []) or genre_families(raw_genre)
    lyrics = row.get("lyrics")
    year = row.get("year")
    extra = _parse_extra(row)

    bpm = row.get("bpm")
    energy = row.get("energy")
    valence = row.get("valence")
    dance = row.get("danceability")
    bass_ratio = extra.get("bass_ratio")
    dissonance = extra.get("dissonance")
    acoustic_proxy = extra.get("acoustic_proxy")

    subgenres: list[str] = []
    if raw_genre:
        for part in re.split(r"[,;/|]+", raw_genre):
            norm = normalize_tag(part)
            if norm and norm not in subgenres:
                subgenres.append(norm)
    for fam in families:
        if fam not in subgenres:
            subgenres.append(fam)

    moods: list[str] = []
    textures: list[str] = []
    scenes: list[str] = []

    if energy is not None:
        e = float(energy)
        if e >= 0.72:
            moods.extend(["energetic", "intense"])
            scenes.append("workout")
        elif e <= 0.42:
            moods.extend(["calm", "chill"])
            scenes.extend(["evening", "focus"])

    if valence is not None:
        v = float(valence)
        if v <= 0.36:
            moods.extend(["dark", "melancholic"])
        elif v >= 0.64:
            moods.extend(["uplifting", "bright"])

    if dance is not None and bpm is not None and float(dance) >= 0.58 and float(bpm) >= 105:
        moods.append("danceable")
        scenes.append("party")

    if bass_ratio is not None and float(bass_ratio) >= 0.45:
        textures.append("heavy_bass")
    if dissonance is not None and float(dissonance) >= 0.46:
        moods.append("tense")
    if acoustic_proxy is not None and float(acoustic_proxy) >= 0.60:
        textures.extend(["acoustic", "acoustic_guitar"])

    lang = _detect_language(title, artists, lyrics)
    if "rap" in families and lang == "ru" and "russian_rap" not in subgenres:
        subgenres.insert(0, "russian_rap")
    if "instrumental" in families and lang is None:
        lang = "instrumental"

    era = _era_from_year(year)
    acousticness = float(acoustic_proxy) if acoustic_proxy is not None else None
    instrumentalness = 0.85 if ("instrumental" in families or lang == "instrumental") else None

    summary_parts = []
    if subgenres:
        summary_parts.append(", ".join(subgenres[:4]))
    if moods:
        summary_parts.append(", ".join(dict.fromkeys(moods))[:60])
    if textures:
        summary_parts.append(", ".join(dict.fromkeys(textures))[:60])
    vibe_summary = " | ".join(summary_parts) if summary_parts else None

    return {
        "subgenres": normalize_tag_list(subgenres, limit=8),
        "moods": normalize_tag_list(moods, limit=8),
        "textures": normalize_tag_list(textures, limit=8),
        "scenes": normalize_tag_list(scenes, limit=8),
        "language": lang,
        "era": era,
        "semantic_energy": None,
        "semantic_valence": None,
        "acousticness": round(acousticness, 4) if acousticness is not None else None,
        "instrumentalness": instrumentalness,
        "vibe_summary": vibe_summary,
        "tags_source": "heuristic",
    }


_TAGGER_SYSTEM_PROMPT = """You are an expert music cataloguer and sonic analyst.
Given a batch of music tracks (metadata + DSP summary + optional lyrics snippet), classify each track accurately.
Return STRICT JSON only:
{
  "tracks": [
    {
      "id": <int track_id>,
      "subgenres": ["<canonical snake_case subgenre>"],
      "moods": ["<canonical mood>"],
      "textures": ["<canonical instrument/production texture>"],
      "scenes": ["<canonical scene/activity>"],
      "language": "ru" | "en" | "instrumental" | "other",
      "era": "<e.g. 2010s, 2020s>",
      "energy": <0.0..1.0 real musical arousal/intensity, NOT mastering loudness>,
      "valence": <0.0..1.0 emotional positivity: 0.1=bleak/sad/dark, 0.5=neutral/tense, 0.85=joyful/bright>,
      "acousticness": <0.0..1.0 acoustic instruments vs electronic/synthesized>,
      "instrumentalness": <0.0..1.0>,
      "vibe_summary": "<short bilingual RU+EN summary <=12 words of sound, instruments, mood>"
    }
  ]
}

Use canonical lowercase snake_case tags wherever possible:
- subgenres: russian_rap, trap, rage, drill, phonk, dark_phonk, boom_bap, jazz_rap, acoustic_rap, cloud_rap, abstract_hip_hop, pop_rap, indie_pop, synthpop, dance_pop, art_pop, indie_rock, alt_rock, post_punk, punk, metal, grunge, shoegaze, indie_folk, acoustic_folk, chanson, house, techno, dnb, ambient, synthwave, downtempo, trip_hop, lo_fi_beats, nu_jazz, jazz, blues, rnb, neo_soul, funk, classical, soundtrack.
- moods: dark, aggressive, intense, melancholic, sad, nostalgic, warm, tender, dreamy, atmospheric, calm, chill, uplifting, happy, euphoric, party, romantic, sensual, nocturnal, epic.
- textures: acoustic_guitar, electric_guitar, piano, keys, jazz_samples, saxophone, brass, strings, 808_bass, heavy_bass, synth, pad, live_drums, lo_fi, distorted, clean_vocals, autotune, raspy_vocals, acoustic, orchestral.
- scenes: night_drive, rainy, sunset, road_trip, romantic, workout, running, focus, late_night, morning, party, club, relax, evening.
Be honest about dark/sad songs even in pop or major keys (e.g. Lana Del Rey is melancholic/sad, low valence). Never confuse hard trap/rage with acoustic."""


def _clamp01(value: Any) -> float | None:
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return round(max(0.0, min(1.0, v)), 4)


def _merge_llm_with_heuristic(llm_item: dict[str, Any], base: dict[str, Any]) -> dict[str, Any]:
    subgenres = normalize_tag_list(llm_item.get("subgenres"), limit=8) or base["subgenres"]
    moods = normalize_tag_list(llm_item.get("moods"), limit=8) or base["moods"]
    textures = normalize_tag_list(llm_item.get("textures"), limit=8) or base["textures"]
    scenes = normalize_tag_list(llm_item.get("scenes"), limit=8) or base["scenes"]

    lang = str(llm_item.get("language") or "").strip().lower()
    if lang not in ("ru", "en", "instrumental", "other"):
        lang = base["language"]

    era_raw = str(llm_item.get("era") or "").strip()
    era = era_raw if re.fullmatch(r"\d{4}s", era_raw) else base["era"]

    sem_energy = _clamp01(llm_item.get("energy"))
    sem_valence = _clamp01(llm_item.get("valence"))
    acousticness = _clamp01(llm_item.get("acousticness"))
    if acousticness is None:
        acousticness = base["acousticness"]
    instrumentalness = _clamp01(llm_item.get("instrumentalness"))
    if instrumentalness is None:
        instrumentalness = base["instrumentalness"]

    vibe = str(llm_item.get("vibe_summary") or "").strip()[:300] or base["vibe_summary"]
    return {
        "subgenres": subgenres,
        "moods": moods,
        "textures": textures,
        "scenes": scenes,
        "language": lang,
        "era": era,
        "semantic_energy": sem_energy,
        "semantic_valence": sem_valence,
        "acousticness": acousticness,
        "instrumentalness": instrumentalness,
        "vibe_summary": vibe,
        "tags_source": "llm",
    }


def _track_prompt_item(row: dict[str, Any]) -> dict[str, Any]:
    extra = _parse_extra(row)
    item: dict[str, Any] = {
        "id": int(row["track_id"]),
        "title": row.get("title") or "",
        "artists": list(row.get("artists") or [])[:4],
        "album": row.get("album") or "",
        "genre": row.get("genre") or "",
        "year": row.get("year"),
    }
    dsp: dict[str, Any] = {}
    if row.get("bpm") is not None:
        dsp["bpm"] = round(float(row["bpm"]), 1)
    if extra.get("dsp_energy") is not None:
        dsp["energy"] = round(float(extra["dsp_energy"]), 2)
    elif row.get("energy") is not None:
        dsp["energy"] = round(float(row["energy"]), 2)
    if extra.get("bass_ratio") is not None:
        dsp["bass"] = round(float(extra["bass_ratio"]), 2)
    if extra.get("acoustic_proxy") is not None:
        dsp["acoustic_proxy"] = round(float(extra["acoustic_proxy"]), 2)
    if dsp:
        item["dsp"] = dsp
    lyrics = (row.get("lyrics") or "").strip()
    if lyrics:
        item["lyrics_excerpt"] = re.sub(r"\s+", " ", lyrics)[:280]
    return item


async def tag_batch(
    rows: list[dict[str, Any]],
    *,
    api_key: str | None = None,
    endpoint: str | None = None,
    model: str | None = None,
) -> dict[int, dict[str, Any]]:
    """Tag a batch of track rows via LLM with automatic heuristic fallback."""
    heuristics: dict[int, dict[str, Any]] = {
        int(r["track_id"]): heuristic_tags(r) for r in rows
    }
    key = (api_key or "").strip() or settings.deepseek_api_key
    base_url = (endpoint or "").strip() or settings.deepseek_base_url
    mdl = (model or "").strip() or settings.deepseek_model
    if not key or not rows:
        return heuristics

    payload_items = [_track_prompt_item(r) for r in rows]
    body = {
        "model": mdl,
        "messages": [
            {"role": "system", "content": _TAGGER_SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps({"tracks": payload_items}, ensure_ascii=False)},
        ],
        "temperature": 0.15,
        "max_tokens": min(4096, max(1200, 320 * len(rows) + 250)),
        "response_format": {"type": "json_object"},
    }
    try:
        async with httpx.AsyncClient(timeout=max(45.0, settings.deepseek_timeout)) as client:
            resp = await client.post(
                f"{base_url.rstrip('/')}/chat/completions",
                headers={"Authorization": f"Bearer {key}"},
                json=body,
            )
            resp.raise_for_status()
            raw_content = resp.json()["choices"][0]["message"]["content"]
        parsed = json.loads(raw_content)
        out = dict(heuristics)
        for item in parsed.get("tracks") or []:
            if not isinstance(item, dict) or item.get("id") is None:
                continue
            try:
                tid = int(item["id"])
            except (TypeError, ValueError):
                continue
            if tid in out:
                out[tid] = _merge_llm_with_heuristic(item, out[tid])
        return out
    except Exception:
        log.exception("LLM batch tagging failed; falling back to heuristic tags")
        return heuristics


def build_tags_json(tagged: dict[str, Any]) -> dict[str, Any]:
    """Compact JSONB representation for `track_features.tags`."""
    out: dict[str, Any] = {}
    for field in ("subgenres", "moods", "textures", "scenes"):
        if tagged.get(field):
            out[field] = tagged[field]
    if tagged.get("language"):
        out["language"] = tagged["language"]
    if tagged.get("era"):
        out["era"] = tagged["era"]
    if tagged.get("vibe_summary"):
        out["vibe_summary"] = tagged["vibe_summary"]
    if tagged.get("tags_source"):
        out["source"] = tagged["tags_source"]
    return out
