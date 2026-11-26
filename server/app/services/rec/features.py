"""Audio-feature vectors and corpus Z-score normalization for the recommender.

Turns a `track_features` row (Essentia DSP columns + `dsp_extra` + MFCC[1..4])
into an L2-normalized, Z-score standardized 22-dimensional vector so cosine
similarity across `track_emb_audio` and taste centroids accurately separates
timbre, rhythm, bass weight, brightness, and dynamics.
"""

from __future__ import annotations

import json
import math
from typing import Any

# Ordered feature specs: (name, fallback_mean, fallback_std)
FEATURE_SPECS: tuple[tuple[str, float, float], ...] = (
    ("loudness_lufs", -10.5, 4.5),
    ("dynamic_complexity", 4.2, 2.0),
    ("bpm", 120.0, 26.0),
    ("beat_confidence", 0.65, 0.35),
    ("danceability", 0.45, 0.16),
    ("onset_rate", 3.5, 1.8),
    ("spectral_centroid", 2000.0, 900.0),
    ("spectral_rolloff", 4200.0, 2000.0),
    ("spectral_flux_log", 1.2, 0.8),
    ("spectral_flatness", 0.08, 0.08),
    ("spectral_contrast", 0.55, 0.18),
    ("zero_crossing_rate", 0.06, 0.04),
    ("dissonance", 0.40, 0.12),
    ("pitch_salience", 0.55, 0.15),
    ("bass_ratio", 0.35, 0.18),
    ("mid_ratio", 0.45, 0.15),
    ("energy", 0.55, 0.22),
    ("valence", 0.48, 0.22),
    ("mfcc_1", 45.0, 30.0),
    ("mfcc_2", -10.0, 22.0),
    ("mfcc_3", 10.0, 16.0),
    ("mfcc_4", -5.0, 14.0),
)

MIN_KNOWN = 4
STATS_ID = "dsp_zscore_v2"
_Z_CLIP = 3.5

_corpus_stats_cache: dict[str, tuple[float, float]] | None = None


def _parse_json_dict(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {}
        except ValueError:
            return {}
    return {}


def extract_raw_dims(row: dict[str, Any] | None) -> dict[str, float]:
    """Extract raw numeric DSP values from a track_features row."""
    if not row:
        return {}
    extra = _parse_json_dict(row.get("dsp_extra"))
    out: dict[str, float] = {}

    def _put(name: str, val: Any) -> None:
        if val is None:
            return
        try:
            fval = float(val)
        except (TypeError, ValueError):
            return
        if math.isfinite(fval):
            out[name] = fval

    for col in (
        "loudness_lufs",
        "dynamic_complexity",
        "bpm",
        "beat_confidence",
        "danceability",
        "onset_rate",
        "spectral_centroid",
        "spectral_rolloff",
        "energy",
        "valence",
    ):
        _put(col, row.get(col))

    flux = row.get("spectral_flux")
    if flux is not None:
        try:
            f_flux = float(flux)
            if math.isfinite(f_flux):
                out["spectral_flux_log"] = math.log1p(max(0.0, f_flux))
        except (TypeError, ValueError):
            pass

    for extra_key in (
        "spectral_flatness",
        "spectral_contrast",
        "zero_crossing_rate",
        "dissonance",
        "pitch_salience",
        "bass_ratio",
        "mid_ratio",
    ):
        _put(extra_key, extra.get(extra_key))

    mfcc = row.get("mfcc_mean") or []
    # Skip mfcc[0] (overall frame log energy, redundant with loudness_lufs) and use timbre 1..4.
    for idx in range(1, 5):
        if idx < len(mfcc):
            _put(f"mfcc_{idx}", mfcc[idx])

    return out


def default_stats() -> dict[str, tuple[float, float]]:
    return {name: (mean, std) for name, mean, std in FEATURE_SPECS}


def set_cached_stats(stats: dict[str, tuple[float, float]] | None) -> None:
    global _corpus_stats_cache
    _corpus_stats_cache = stats


def get_cached_stats() -> dict[str, tuple[float, float]]:
    return _corpus_stats_cache if _corpus_stats_cache is not None else default_stats()


async def load_corpus_stats() -> dict[str, tuple[float, float]]:
    """Load Z-score stats from `rec_feature_stats`, falling back to domain priors."""
    from ... import db

    row = await db.fetch_one("SELECT stats FROM rec_feature_stats WHERE id = :i", {"i": STATS_ID})
    if not row or not row.get("stats"):
        return default_stats()
    raw = _parse_json_dict(row["stats"])
    means = raw.get("means") or {}
    stds = raw.get("stds") or {}
    merged = default_stats()
    for name, _, _ in FEATURE_SPECS:
        if name in means and name in stds:
            try:
                m = float(means[name])
                s = float(stds[name])
                if math.isfinite(m) and math.isfinite(s) and s > 1e-6:
                    merged[name] = (m, s)
            except (TypeError, ValueError):
                continue
    set_cached_stats(merged)
    return merged


async def recompute_corpus_stats(min_rows: int = 25) -> dict[str, tuple[float, float]]:
    """Compute corpus mean/std across all indexed tracks and persist to Postgres."""
    from ... import db

    rows = await db.fetch_all(
        """
        SELECT loudness_lufs, dynamic_complexity, bpm, beat_confidence, danceability,
               onset_rate, spectral_centroid, spectral_rolloff, spectral_flux,
               energy, valence, mfcc_mean, dsp_extra
        FROM track_features
        """
    )
    buckets: dict[str, list[float]] = {name: [] for name, _, _ in FEATURE_SPECS}
    for row in rows:
        dims = extract_raw_dims(row)
        for name, val in dims.items():
            if name in buckets:
                buckets[name].append(val)

    stats = default_stats()
    means_json: dict[str, float] = {}
    stds_json: dict[str, float] = {}
    for name, fb_mean, fb_std in FEATURE_SPECS:
        vals = buckets[name]
        if len(vals) >= min_rows:
            mean = sum(vals) / len(vals)
            var = sum((v - mean) ** 2 for v in vals) / len(vals)
            std = math.sqrt(var)
            if std > 1e-6:
                stats[name] = (mean, std)
                means_json[name] = round(mean, 6)
                stds_json[name] = round(std, 6)
                continue
        means_json[name] = fb_mean
        stds_json[name] = fb_std

    payload = {"count": len(rows), "means": means_json, "stds": stds_json}
    await db.execute(
        """
        INSERT INTO rec_feature_stats (id, stats, updated_at)
        VALUES (:i, CAST(:s AS jsonb), NOW())
        ON CONFLICT (id) DO UPDATE SET stats = EXCLUDED.stats, updated_at = NOW()
        """,
        {"i": STATS_ID, "s": json.dumps(payload)},
    )
    set_cached_stats(stats)
    return stats


def feature_vector(
    row: dict[str, Any] | None,
    stats: dict[str, tuple[float, float]] | None = None,
) -> list[float] | None:
    """Z-score standardized, L2-normalized audio vector; None if sparse."""
    dims = extract_raw_dims(row)
    if len(dims) < MIN_KNOWN:
        return None
    active_stats = stats if stats is not None else get_cached_stats()

    vec: list[float] = []
    for name, fb_mean, fb_std in FEATURE_SPECS:
        if name not in dims:
            vec.append(0.0)
            continue
        mean, std = active_stats.get(name, (fb_mean, fb_std))
        std = std if std > 1e-6 else fb_std
        z = (dims[name] - mean) / std
        z = max(-_Z_CLIP, min(_Z_CLIP, z))
        vec.append(z)

    norm = math.sqrt(sum(v * v for v in vec))
    if norm <= 1e-9:
        return None
    return [round(v / norm, 6) for v in vec]


def cosine(a: list[float] | None, b: list[float] | None) -> float | None:
    """Cosine similarity in [-1, 1]; None when it cannot be computed."""
    if not a or not b or len(a) != len(b):
        return None
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return None
    return dot / (norm_a * norm_b)
