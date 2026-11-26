"""Feature indexer: claims `index_jobs` and fills feature, semantic tag, and embedding tables.

Pipeline for local tracks (`connector = 'local'`):
  1. `dsp`   -> Essentia C++ full DSP extraction + embedded lyrics (`track_features`)
  2. `tags`  -> LLM batch semantic tagging + heuristic fallback (`track_features.*`)
  3. `audio` -> Corpus Z-scored 22-d DSP vector -> `track_emb_audio` (HNSW)
  4. `text`  -> Rich semantic document -> Giga-Embeddings -> `track_emb_text` (HNSW)

Run once after enqueueing every local track:

    python -m app.indexer --enqueue --once

Or as a long-running worker (the compose `indexer` service):

    python -m app.indexer
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import multiprocessing
import os
import re
import shutil
import subprocess
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path
from typing import Any

import essentia.standard as es
import httpx
import mutagen
import numpy as np

from . import db
from .config import settings
from .services.rec import features as rec_features
from .services.rec import tagger as rec_tagger
from .services.rec.audio_embed import get_audio_backend, pad_to_column
from .services.rec.queue import MODEL_VERSIONS, enqueue_track

MODEL_VERSION = MODEL_VERSIONS["dsp"]
TAGS_MODEL_VERSION = MODEL_VERSIONS["tags"]
TEXT_MODEL_VERSION = MODEL_VERSIONS["text"]
AUDIO_MODEL_VERSION = MODEL_VERSIONS["audio"]
POLL_SECONDS = 15
ENQUEUE_INTERVAL = 600
SAMPLE_RATE = 44100
FRAME_SIZE = 2048
HOP_SIZE = 1024
KEY_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
_LRC_TS_RE = re.compile(r"\[\d+:\d+(?:[.:]\d+)?\]")

log = logging.getLogger("kwhale.indexer")

FFMPEG_TIMEOUT_SEC = 600


class AudioDecodeError(RuntimeError):
    """The DSP pipeline cannot read this audio file."""


class CorruptAudio(AudioDecodeError):
    """Neither Essentia nor ffmpeg could decode the file (truncated/corrupt)."""


class DspWorkerCrash(AudioDecodeError):
    """The isolated Essentia worker died (native heap bug on this input)."""


class NoLocalSource(AudioDecodeError):
    """The track no longer has a local file (merged/removed): drop the job."""


class FeaturesNotReady(RuntimeError):
    """The DSP row the audio backend builds on has not landed yet (retry later)."""


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _extract_embedded_lyrics(path: str) -> str | None:
    """Read plain/synced lyrics embedded inside local audio file tags via mutagen."""
    try:
        mf = mutagen.File(path)
    except Exception:
        return None
    if mf is None or not getattr(mf, "tags", None):
        return None
    tags = mf.tags
    raw_chunks: list[str] = []

    try:
        for key in tags.keys():
            k_low = str(key).lower()
            if k_low.startswith("uslt") or k_low in ("lyrics", "©lyr", "\xa9lyr", "unsyncedlyrics"):
                val = tags[key]
                if hasattr(val, "text"):
                    txt = val.text
                    raw_chunks.append("\n".join(txt) if isinstance(txt, list) else str(txt))
                elif isinstance(val, list):
                    raw_chunks.append("\n".join(str(v) for v in val))
                else:
                    raw_chunks.append(str(val))
    except Exception:
        return None

    for chunk in raw_chunks:
        cleaned = _LRC_TS_RE.sub("", chunk or "")
        cleaned = re.sub(r"[ \t]+", " ", cleaned)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
        if cleaned:
            return cleaned[:4000]
    return None


def _load_audio(path: str):
    """Load audio with Essentia, falling back to an ffmpeg-decoded WAV."""
    try:
        return es.AudioLoader(filename=path)()
    except Exception as direct_exc:  # noqa: BLE001 — fall back for any decode error
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            raise AudioDecodeError(
                f"AudioLoader failed ({direct_exc}); ffmpeg is not installed"
            ) from direct_exc

        fd, wav_path = tempfile.mkstemp(prefix="kwhale-dsp-", suffix=".wav")
        os.close(fd)
        try:
            proc = subprocess.run(  # noqa: S603 — fixed argv, path is a local file
                [
                    ffmpeg,
                    "-v",
                    "error",
                    "-y",
                    "-i",
                    path,
                    "-vn",
                    "-ac",
                    "2",
                    "-ar",
                    str(SAMPLE_RATE),
                    "-f",
                    "wav",
                    wav_path,
                ],
                capture_output=True,
                text=True,
                timeout=FFMPEG_TIMEOUT_SEC,
                check=False,
            )
            if proc.returncode != 0 or os.path.getsize(wav_path) == 0:
                tail = (proc.stderr or "").strip().splitlines()
                reason = tail[-1][:200] if tail else "unknown ffmpeg error"
                raise CorruptAudio(f"ffmpeg decode failed: {reason}") from direct_exc
            try:
                return es.AudioLoader(filename=wav_path)()
            except Exception as wav_exc:  # noqa: BLE001
                raise AudioDecodeError(
                    f"AudioLoader failed after ffmpeg transcode: {wav_exc}"
                ) from wav_exc
        except subprocess.TimeoutExpired as exc:
            raise CorruptAudio("ffmpeg decode timed out") from exc
        finally:
            try:
                os.unlink(wav_path)
            except OSError:
                pass


def extract_dsp(path: str) -> dict[str, Any]:
    """Extract comprehensive C++ Essentia DSP features + embedded lyrics."""
    loaded = _load_audio(path)
    stereo = np.asarray(loaded[0], dtype=np.float32)
    sample_rate = float(loaded[1])
    mono = stereo.mean(axis=1) if stereo.ndim > 1 else stereo
    if int(round(sample_rate)) != SAMPLE_RATE and len(mono) > 0:
        mono = es.Resample(inputSampleRate=sample_rate, outputSampleRate=float(SAMPLE_RATE))(mono)
        work_sr = float(SAMPLE_RATE)
    else:
        work_sr = sample_rate

    bpm, _beats, beat_confidence, _estimates, _intervals = es.RhythmExtractor2013(
        method="multifeature"
    )(mono)
    key, scale, key_strength = es.KeyExtractor()(mono)
    dance_result = es.Danceability()(mono)
    danceability = float(dance_result[0] if isinstance(dance_result, tuple) else dance_result)
    danceability = _clamp01(danceability / 3.0)

    try:
        integrated = float(es.LoudnessEBUR128(sampleRate=sample_rate)(stereo)[2])
    except Exception:
        integrated = None

    try:
        dyn_comp_res = es.DynamicComplexity()(mono)
        dynamic_complexity = float(
            dyn_comp_res[0] if isinstance(dyn_comp_res, tuple) else dyn_comp_res
        )
    except Exception:
        dynamic_complexity = None

    try:
        _onsets, onset_rate_val = es.OnsetRate()(mono)
        onset_rate = float(onset_rate_val)
    except Exception:
        onset_rate = None

    windowing = es.Windowing(type="hann")
    spectrum = es.Spectrum()
    mfcc = es.MFCC(numberCoefficients=13)
    centroid_alg = es.Centroid(range=work_sr / 2.0)
    rolloff_alg = es.RollOff(sampleRate=work_sr)
    flux_alg = es.Flux()
    flatness_alg = es.Flatness()
    zcr_alg = es.ZeroCrossingRate()
    contrast_alg = es.SpectralContrast(frameSize=FRAME_SIZE, sampleRate=work_sr)
    peaks_alg = es.SpectralPeaks(sampleRate=work_sr)
    dissonance_alg = es.Dissonance()
    salience_alg = es.PitchSalience(sampleRate=work_sr)

    nyquist = work_sr / 2.0
    n_bins = FRAME_SIZE // 2 + 1
    freqs = np.linspace(0.0, nyquist, n_bins)
    bass_mask = (freqs >= 20.0) & (freqs <= 160.0)
    mid_mask = (freqs >= 300.0) & (freqs <= 3400.0)
    high_mask = freqs >= 4000.0

    mfcc_rows: list[list[float]] = []
    centroids: list[float] = []
    rolloffs: list[float] = []
    fluxes: list[float] = []
    flatnesses: list[float] = []
    zcrs: list[float] = []
    contrasts: list[float] = []
    dissonances: list[float] = []
    saliences: list[float] = []
    bass_ratios: list[float] = []
    mid_ratios: list[float] = []
    high_ratios: list[float] = []

    for frame in es.FrameGenerator(mono, frameSize=FRAME_SIZE, hopSize=HOP_SIZE):
        windowed = windowing(frame)
        spec = spectrum(windowed)
        # Essentia MFCC returns (mel_bands, mfcc_coefficients)
        _bands, mfcc_coeffs = mfcc(spec)
        mfcc_rows.append([float(v) for v in mfcc_coeffs])
        centroids.append(float(centroid_alg(spec)))
        rolloffs.append(float(rolloff_alg(spec)))
        fluxes.append(float(flux_alg(spec)))
        flatnesses.append(float(flatness_alg(spec)))
        zcrs.append(float(zcr_alg(frame)))

        try:
            c_vals, _valleys = contrast_alg(spec)
            contrasts.append(float(np.mean(c_vals)))
        except Exception:
            pass
        try:
            sal = float(salience_alg(spec))
            if math.isfinite(sal):
                saliences.append(sal)
        except Exception:
            pass
        try:
            p_freqs, p_mags = peaks_alg(spec)
            if len(p_freqs) > 1 and p_freqs[0] > 0.0:
                dis = float(dissonance_alg(p_freqs, p_mags))
                if math.isfinite(dis):
                    dissonances.append(dis)
        except Exception:
            pass

        power = np.asarray(spec, dtype=np.float32) ** 2
        total_e = float(np.sum(power))
        if total_e > 1e-12:
            bass_ratios.append(float(np.sum(power[bass_mask])) / total_e)
            mid_ratios.append(float(np.sum(power[mid_mask])) / total_e)
            high_ratios.append(float(np.sum(power[high_mask])) / total_e)

    mfcc_array = np.asarray(mfcc_rows, dtype=np.float32) if mfcc_rows else np.empty((0, 13))
    mfcc_mean = mfcc_array.mean(axis=0).round(4).tolist() if len(mfcc_rows) else None
    mfcc_std = mfcc_array.std(axis=0).round(4).tolist() if len(mfcc_rows) else None

    bpm_value = float(bpm)
    mode_val = 1 if scale == "major" else 0
    key_str = float(key_strength)
    dance_val = round(danceability, 4)

    mean_centroid = float(np.mean(centroids)) if centroids else None
    mean_rolloff = float(np.mean(rolloffs)) if rolloffs else None
    mean_flux = float(np.mean(fluxes)) if fluxes else None
    mean_flatness = float(np.mean(flatnesses)) if flatnesses else None
    mean_zcr = float(np.mean(zcrs)) if zcrs else None
    mean_contrast = float(np.mean(contrasts)) if contrasts else None
    mean_dissonance = float(np.mean(dissonances)) if dissonances else None
    mean_salience = float(np.mean(saliences)) if saliences else None
    mean_bass = float(np.mean(bass_ratios)) if bass_ratios else None
    mean_mid = float(np.mean(mid_ratios)) if mid_ratios else None
    mean_high = float(np.mean(high_ratios)) if high_ratios else None

    # Calibrated musical energy (arousal): note density + spectral change + tempo + drive + gentle LUFS.
    # Avoids treating loud-mastered acoustic ballads as high-energy tracks.
    loud_norm = _clamp01((integrated + 32.0) / 26.0) if integrated is not None else 0.5
    onset_norm = _clamp01(onset_rate / 7.5) if onset_rate is not None else 0.45
    flux_norm = _clamp01(math.log1p(max(0.0, mean_flux)) / 2.2) if mean_flux is not None else 0.45
    bpm_norm = _clamp01((bpm_value - 60.0) / 110.0) if bpm_value > 0 else 0.45
    zcr_norm = _clamp01(mean_zcr / 0.14) if mean_zcr is not None else 0.4

    dsp_energy = round(
        _clamp01(
            0.30 * onset_norm
            + 0.22 * flux_norm
            + 0.16 * bpm_norm
            + 0.12 * zcr_norm
            + 0.10 * dance_val
            + 0.10 * loud_norm
        ),
        4,
    )

    # Calibrated DSP valence (used as baseline; blended with LLM semantic valence when tagged).
    mode_delta = (0.14 if mode_val == 1 else -0.14) * _clamp01((key_str - 0.35) / 0.45)
    bright_norm = _clamp01((mean_centroid - 800.0) / 2800.0) if mean_centroid is not None else 0.5
    diss_norm = _clamp01((mean_dissonance or 0.4) / 0.7)
    dsp_valence = round(
        _clamp01(
            0.48
            + mode_delta
            + 0.22 * (dance_val - 0.5)
            + 0.18 * (bright_norm - 0.5)
            - 0.22 * (diss_norm - 0.5)
            + 0.12 * (dsp_energy - 0.5)
        ),
        4,
    )

    # Acousticness proxy: dynamic range + spectral contrast - electronic flatness - sub-bass overload.
    dyn_norm = _clamp01((dynamic_complexity or 4.0) / 9.0)
    contrast_norm = _clamp01(((mean_contrast or 0.5) - 0.2) / 0.6)
    flat_penalty = _clamp01((mean_flatness or 0.08) / 0.25)
    bass_penalty = _clamp01(((mean_bass or 0.3) - 0.25) / 0.45)
    acoustic_proxy = round(
        _clamp01(0.35 * dyn_norm + 0.35 * contrast_norm + 0.15 * (1.0 - flat_penalty) + 0.15 * (1.0 - bass_penalty)),
        4,
    )

    dsp_extra = {
        "key_strength": round(key_str, 4),
        "spectral_flatness": round(mean_flatness, 5) if mean_flatness is not None else None,
        "spectral_contrast": round(mean_contrast, 4) if mean_contrast is not None else None,
        "zero_crossing_rate": round(mean_zcr, 5) if mean_zcr is not None else None,
        "dissonance": round(mean_dissonance, 4) if mean_dissonance is not None else None,
        "pitch_salience": round(mean_salience, 4) if mean_salience is not None else None,
        "bass_ratio": round(mean_bass, 4) if mean_bass is not None else None,
        "mid_ratio": round(mean_mid, 4) if mean_mid is not None else None,
        "high_ratio": round(mean_high, 4) if mean_high is not None else None,
        "acoustic_proxy": acoustic_proxy,
        "dsp_energy": dsp_energy,
        "dsp_valence": dsp_valence,
    }
    dsp_extra = {k: v for k, v in dsp_extra.items() if v is not None}

    lyrics = _extract_embedded_lyrics(path)

    return {
        "bpm": round(bpm_value, 2) if 0.0 < bpm_value <= 300.0 else None,
        "beat_confidence": round(float(beat_confidence), 4),
        "key": KEY_NAMES.index(key) if key in KEY_NAMES else None,
        "mode": mode_val,
        "loudness_lufs": round(integrated, 2) if integrated is not None else None,
        "dynamic_complexity": round(dynamic_complexity, 4) if dynamic_complexity is not None else None,
        "danceability": dance_val,
        "onset_rate": round(onset_rate, 4) if onset_rate is not None else None,
        "spectral_centroid": round(mean_centroid, 2) if mean_centroid is not None else None,
        "spectral_rolloff": round(mean_rolloff, 2) if mean_rolloff is not None else None,
        "spectral_flux": round(mean_flux, 5) if mean_flux is not None else None,
        "mfcc_mean": mfcc_mean,
        "mfcc_std": mfcc_std,
        "energy": dsp_energy,
        "valence": dsp_valence,
        "dsp_extra": json.dumps(dsp_extra),
        "lyrics": lyrics,
    }


_pool: ProcessPoolExecutor | None = None
_DSP_WORKERS = max(1, min(4, int(os.environ.get("INDEXER_DSP_WORKERS", "2") or 2)))


def _get_pool() -> ProcessPoolExecutor:
    global _pool
    if _pool is None:
        _pool = ProcessPoolExecutor(
            max_workers=_DSP_WORKERS,
            mp_context=multiprocessing.get_context("spawn"),
        )
    return _pool


def _reset_pool() -> None:
    global _pool
    if _pool is not None:
        _pool.shutdown(wait=False, cancel_futures=True)
    _pool = None


async def _extract_dsp_isolated(path: str) -> dict[str, Any]:
    loop = asyncio.get_running_loop()
    try:
        return await loop.run_in_executor(_get_pool(), extract_dsp, path)
    except BrokenProcessPool as exc:
        _reset_pool()
        raise DspWorkerCrash(f"Essentia worker crashed on {path}") from exc


async def enqueue_local() -> int:
    await db.execute(
        "UPDATE index_jobs SET status = 'pending', last_error = NULL "
        "WHERE kind = 'dsp' AND model_version = :v AND status = 'failed' AND attempts < 5",
        {"v": MODEL_VERSION},
    )
    rows = await db.execute_returning(
        """
        INSERT INTO index_jobs (track_id, kind, model_version)
        SELECT DISTINCT s.track_id, 'dsp', :v
        FROM track_sources s
        WHERE s.connector = 'local' AND s.file_path IS NOT NULL
        ON CONFLICT (track_id, kind, model_version) DO NOTHING
        RETURNING id
        """,
        {"v": MODEL_VERSION},
    )
    return len(rows)


async def enqueue_tags() -> int:
    """Queue semantic tagging (`kind='tags'`) for every local track that lacks current tags."""
    await db.execute(
        "UPDATE index_jobs SET status = 'pending', last_error = NULL "
        "WHERE kind = 'tags' AND model_version = :v AND status = 'failed' AND attempts < 5",
        {"v": TAGS_MODEL_VERSION},
    )
    rows = await db.execute_returning(
        """
        INSERT INTO index_jobs (track_id, kind, model_version)
        SELECT DISTINCT s.track_id, 'tags', :v
        FROM track_sources s
        LEFT JOIN track_features tf ON tf.track_id = s.track_id
        WHERE s.connector = 'local' AND s.file_path IS NOT NULL
          AND (tf.track_id IS NULL OR tf.tags_version IS DISTINCT FROM :v)
        ON CONFLICT (track_id, kind, model_version) DO NOTHING
        RETURNING id
        """,
        {"v": TAGS_MODEL_VERSION},
    )
    return len(rows)


async def enqueue_text() -> int:
    """Queue Giga text embeddings (`giga-v2-rich`) for local tracks lacking them."""
    rows = await db.execute_returning(
        """
        INSERT INTO index_jobs (track_id, kind, model_version)
        SELECT DISTINCT s.track_id, 'text', :v
        FROM track_sources s
        LEFT JOIN track_emb_text e ON e.track_id = s.track_id
        WHERE s.connector = 'local' AND s.file_path IS NOT NULL
          AND (e.track_id IS NULL OR e.model_version IS DISTINCT FROM :v)
        ON CONFLICT (track_id, kind, model_version) DO NOTHING
        RETURNING id
        """,
        {"v": TEXT_MODEL_VERSION},
    )
    return len(rows)


async def enqueue_audio() -> int:
    """Queue audio embeddings for every local track that lacks current Z-scored vector."""
    await db.execute(
        "UPDATE index_jobs SET status = 'pending', last_error = NULL "
        "WHERE kind = 'audio' AND model_version = :v AND status = 'failed' AND attempts < 5",
        {"v": AUDIO_MODEL_VERSION},
    )
    rows = await db.execute_returning(
        """
        INSERT INTO index_jobs (track_id, kind, model_version)
        SELECT DISTINCT s.track_id, 'audio', :v
        FROM track_sources s
        WHERE s.connector = 'local' AND s.file_path IS NOT NULL
          AND NOT EXISTS (
              SELECT 1 FROM track_emb_audio e
              WHERE e.track_id = s.track_id AND e.model_version = :v
          )
        ON CONFLICT (track_id, kind, model_version) DO NOTHING
        RETURNING id
        """,
        {"v": AUDIO_MODEL_VERSION},
    )
    return len(rows)


async def claim(kinds: tuple[str, ...] = ("dsp", "text")) -> dict | None:
    rows = await db.execute_returning(
        """
        UPDATE index_jobs
        SET status = 'running', updated_at = NOW()
        WHERE id = (
            SELECT j.id FROM index_jobs j
            WHERE j.status = 'pending' AND j.kind = ANY(:kinds)
              AND (
                  (j.kind = 'dsp' AND j.model_version = :vd)
                  OR (j.kind = 'text' AND j.model_version = :vt)
              )
            ORDER BY
                CASE j.kind WHEN 'dsp' THEN 0 ELSE 1 END,
                EXISTS (
                    SELECT 1 FROM track_sources s
                    WHERE s.track_id = j.track_id AND s.connector = 'local' AND s.available
                ) DESC,
                j.id
            LIMIT 1
            FOR UPDATE SKIP LOCKED
        )
        RETURNING id, track_id, kind
        """,
        {"kinds": list(kinds), "vd": MODEL_VERSION, "vt": TEXT_MODEL_VERSION},
    )
    return rows[0] if rows else None


async def claim_tags_batch(limit: int = 20) -> list[dict]:
    """Claim pending `tags` jobs, preferring tracks that already have DSP features."""
    return await db.execute_returning(
        """
        UPDATE index_jobs
        SET status = 'running', updated_at = NOW()
        WHERE id IN (
            SELECT j.id FROM index_jobs j
            WHERE j.status = 'pending' AND j.kind = 'tags' AND j.model_version = :v
            ORDER BY EXISTS (
                SELECT 1 FROM track_features tf
                WHERE tf.track_id = j.track_id AND tf.extractor_version IS NOT NULL
            ) DESC, j.id
            LIMIT :n
            FOR UPDATE OF j SKIP LOCKED
        )
        RETURNING id, track_id, kind
        """,
        {"n": limit, "v": TAGS_MODEL_VERSION},
    )


async def claim_audio_batch(limit: int) -> list[dict]:
    """Claim up to `limit` pending audio jobs in one round-trip."""
    return await db.execute_returning(
        """
        UPDATE index_jobs
        SET status = 'running', updated_at = NOW()
        WHERE id IN (
            SELECT id FROM index_jobs
            WHERE status = 'pending' AND kind = 'audio' AND model_version = :v
            ORDER BY id
            LIMIT :n
            FOR UPDATE SKIP LOCKED
        )
        RETURNING id, track_id, kind
        """,
        {"n": limit, "v": AUDIO_MODEL_VERSION},
    )


async def _file_path(track_id: int) -> str | None:
    row = await db.fetch_one(
        """
        SELECT file_path FROM track_sources
        WHERE track_id = :t AND connector = 'local' AND file_path IS NOT NULL
        ORDER BY quality_tier DESC
        LIMIT 1
        """,
        {"t": track_id},
    )
    return (row or {}).get("file_path")


def _describe_dsp_words(row: dict[str, Any]) -> str:
    bits: list[str] = []
    bpm = row.get("bpm")
    if bpm is not None:
        bits.append(f"{round(float(bpm))} BPM")
    energy = row.get("energy")
    if energy is not None:
        e = float(energy)
        bits.append("высокая энергия" if e >= 0.7 else ("спокойный темп" if e <= 0.4 else "средняя энергия"))
    valence = row.get("valence")
    if valence is not None:
        v = float(valence)
        bits.append("светлое настроение" if v >= 0.62 else ("тёмное/меланхоличное настроение" if v <= 0.38 else "нейтральное настроение"))
    return ", ".join(bits)


async def _text_input(track_id: int) -> str | None:
    row = await db.fetch_one(
        """
        SELECT t.title, t.artists, t.album, t.genre, t.year,
               tf.bpm, tf.energy, tf.valence,
               tf.subgenres, tf.moods, tf.textures, tf.scenes,
               tf.language, tf.era, tf.vibe_summary, tf.tags, tf.lyrics
        FROM tracks t
        LEFT JOIN track_features tf ON tf.track_id = t.id
        WHERE t.id = :t
        """,
        {"t": track_id},
    )
    if not row:
        return None
    parts: list[str] = []
    title = (row.get("title") or "").strip()
    artists = [str(a).strip() for a in (row.get("artists") or []) if str(a).strip()]
    if title:
        parts.append(f"Трек: {title}")
    if artists:
        parts.append(f"Исполнитель: {', '.join(artists)}")
    if row.get("album"):
        parts.append(f"Альбом: {row['album']}")
    if row.get("year"):
        era = row.get("era")
        parts.append(f"Год: {row['year']}" + (f" ({era})" if era else ""))
    if row.get("genre"):
        parts.append(f"Жанр: {row['genre']}")
    if row.get("subgenres"):
        parts.append("Поджанры: " + ", ".join(str(x) for x in row["subgenres"]))
    if row.get("moods"):
        parts.append("Настроение: " + ", ".join(str(x) for x in row["moods"]))
    if row.get("textures"):
        parts.append("Звучание и инструменты: " + ", ".join(str(x) for x in row["textures"]))
    if row.get("scenes"):
        parts.append("Сцены: " + ", ".join(str(x) for x in row["scenes"]))
    if row.get("language"):
        parts.append(f"Язык: {row['language']}")
    dsp_words = _describe_dsp_words(row)
    if dsp_words:
        parts.append(f"Аудио: {dsp_words}")
    if row.get("vibe_summary"):
        parts.append(f"Описание: {row['vibe_summary']}")
    tags = row.get("tags") or {}
    if isinstance(tags, dict) and tags and not row.get("subgenres"):
        parts.append("Теги: " + " ".join(str(k) for k in tags))
    if row.get("lyrics"):
        parts.append("Текст: " + str(row["lyrics"])[:1200])
    text = " | ".join(p.strip() for p in parts if p and str(p).strip())
    return text or None


async def embed_text(text: str) -> list[float]:
    async with httpx.AsyncClient(timeout=settings.giga_embeddings_timeout) as client:
        response = await client.post(
            settings.giga_embeddings_url,
            json={"input": [text]},
        )
        response.raise_for_status()
        data = response.json()
    return list(data["data"][0]["embedding"])


async def store_text(track_id: int, vec: list[float]) -> None:
    literal = "[" + ",".join(f"{v:.7f}" for v in vec) + "]"
    await db.execute(
        """
        INSERT INTO track_emb_text (track_id, model_version, vec, indexed_at)
        VALUES (:t, :v, CAST(:vec AS vector), NOW())
        ON CONFLICT (track_id) DO UPDATE SET
            model_version = EXCLUDED.model_version,
            vec = EXCLUDED.vec,
            indexed_at = NOW()
        """,
        {"t": track_id, "v": TEXT_MODEL_VERSION, "vec": literal},
    )


async def store_audio(track_id: int, vec: list[float]) -> None:
    literal = "[" + ",".join(f"{v:.7f}" for v in pad_to_column(vec)) + "]"
    await db.execute(
        """
        INSERT INTO track_emb_audio (track_id, model_version, vec, indexed_at)
        VALUES (:t, :v, CAST(:vec AS vector), NOW())
        ON CONFLICT (track_id) DO UPDATE SET
            model_version = EXCLUDED.model_version,
            vec = EXCLUDED.vec,
            indexed_at = NOW()
        """,
        {"t": track_id, "v": AUDIO_MODEL_VERSION, "vec": literal},
    )


async def store(track_id: int, features: dict[str, Any]) -> None:
    await db.execute(
        """
        INSERT INTO track_features (
            track_id, bpm, beat_confidence, key, mode, loudness_lufs, dynamic_complexity,
            danceability, onset_rate, spectral_centroid, spectral_rolloff, spectral_flux,
            mfcc_mean, mfcc_std, energy, valence, dsp_extra, lyrics,
            extractor_version, indexed_at
        ) VALUES (
            :track_id, :bpm, :beat_confidence, :key, :mode, :loudness_lufs, :dynamic_complexity,
            :danceability, :onset_rate, :spectral_centroid, :spectral_rolloff, :spectral_flux,
            :mfcc_mean, :mfcc_std, :energy, :valence, CAST(:dsp_extra AS jsonb), :lyrics,
            :version, NOW()
        )
        ON CONFLICT (track_id) DO UPDATE SET
            bpm = EXCLUDED.bpm,
            beat_confidence = EXCLUDED.beat_confidence,
            key = EXCLUDED.key,
            mode = EXCLUDED.mode,
            loudness_lufs = EXCLUDED.loudness_lufs,
            dynamic_complexity = EXCLUDED.dynamic_complexity,
            danceability = EXCLUDED.danceability,
            onset_rate = EXCLUDED.onset_rate,
            spectral_centroid = EXCLUDED.spectral_centroid,
            spectral_rolloff = EXCLUDED.spectral_rolloff,
            spectral_flux = EXCLUDED.spectral_flux,
            mfcc_mean = EXCLUDED.mfcc_mean,
            mfcc_std = EXCLUDED.mfcc_std,
            energy = COALESCE(
                CASE WHEN track_features.semantic_energy IS NOT NULL AND EXCLUDED.energy IS NOT NULL
                     THEN 0.55 * track_features.semantic_energy + 0.45 * EXCLUDED.energy
                     ELSE NULL END,
                EXCLUDED.energy
            ),
            valence = COALESCE(
                CASE WHEN track_features.semantic_valence IS NOT NULL AND EXCLUDED.valence IS NOT NULL
                     THEN 0.68 * track_features.semantic_valence + 0.32 * EXCLUDED.valence
                     ELSE NULL END,
                EXCLUDED.valence
            ),
            dsp_extra = EXCLUDED.dsp_extra,
            lyrics = COALESCE(EXCLUDED.lyrics, track_features.lyrics),
            extractor_version = EXCLUDED.extractor_version,
            indexed_at = NOW()
        """,
        {"track_id": track_id, "version": MODEL_VERSION, **features},
    )


async def store_tags(track_id: int, tagged: dict[str, Any]) -> None:
    """Persist semantic tags and calibrated semantic energy/valence onto `track_features`."""
    tags_json = json.dumps(rec_tagger.build_tags_json(tagged), ensure_ascii=False)
    await db.execute(
        """
        INSERT INTO track_features (
            track_id, tags, subgenres, moods, textures, scenes, language, era,
            vibe_summary, semantic_energy, semantic_valence, acousticness,
            instrumentalness, tags_source, tags_version, tags_indexed_at
        ) VALUES (
            :t, CAST(:tags AS jsonb), :subgenres, :moods, :textures, :scenes, :language, :era,
            :vibe_summary, :sem_energy, :sem_valence, :acousticness,
            :instrumentalness, :tags_source, :tags_version, NOW()
        )
        ON CONFLICT (track_id) DO UPDATE SET
            tags = EXCLUDED.tags,
            subgenres = EXCLUDED.subgenres,
            moods = EXCLUDED.moods,
            textures = EXCLUDED.textures,
            scenes = EXCLUDED.scenes,
            language = COALESCE(EXCLUDED.language, track_features.language),
            era = COALESCE(EXCLUDED.era, track_features.era),
            vibe_summary = COALESCE(EXCLUDED.vibe_summary, track_features.vibe_summary),
            semantic_energy = COALESCE(EXCLUDED.semantic_energy, track_features.semantic_energy),
            semantic_valence = COALESCE(EXCLUDED.semantic_valence, track_features.semantic_valence),
            acousticness = COALESCE(EXCLUDED.acousticness, track_features.acousticness),
            instrumentalness = COALESCE(EXCLUDED.instrumentalness, track_features.instrumentalness),
            energy = CASE
                WHEN EXCLUDED.semantic_energy IS NOT NULL AND track_features.energy IS NOT NULL
                    THEN ROUND((0.55 * EXCLUDED.semantic_energy + 0.45 * COALESCE(
                        NULLIF(track_features.dsp_extra->>'dsp_energy', '')::real,
                        track_features.energy
                    ))::numeric, 4)::real
                ELSE COALESCE(EXCLUDED.semantic_energy, track_features.energy)
            END,
            valence = CASE
                WHEN EXCLUDED.semantic_valence IS NOT NULL AND track_features.valence IS NOT NULL
                    THEN ROUND((0.68 * EXCLUDED.semantic_valence + 0.32 * COALESCE(
                        NULLIF(track_features.dsp_extra->>'dsp_valence', '')::real,
                        track_features.valence
                    ))::numeric, 4)::real
                ELSE COALESCE(EXCLUDED.semantic_valence, track_features.valence)
            END,
            tags_source = EXCLUDED.tags_source,
            tags_version = EXCLUDED.tags_version,
            tags_indexed_at = NOW()
        """,
        {
            "t": track_id,
            "tags": tags_json,
            "subgenres": tagged["subgenres"],
            "moods": tagged["moods"],
            "textures": tagged["textures"],
            "scenes": tagged["scenes"],
            "language": tagged.get("language"),
            "era": tagged.get("era"),
            "vibe_summary": tagged.get("vibe_summary"),
            "sem_energy": tagged.get("semantic_energy"),
            "sem_valence": tagged.get("semantic_valence"),
            "acousticness": tagged.get("acousticness"),
            "instrumentalness": tagged.get("instrumentalness"),
            "tags_source": tagged.get("tags_source"),
            "tags_version": TAGS_MODEL_VERSION,
        },
    )


async def _refresh_text_job(track_id: int) -> None:
    """Re-queue `text` embedding after rich tags land so Giga-Embeddings stay in sync."""
    await db.execute(
        """
        INSERT INTO index_jobs (track_id, kind, model_version)
        VALUES (:t, 'text', :v)
        ON CONFLICT (track_id, kind, model_version) DO UPDATE
        SET status = 'pending', attempts = 0, last_error = NULL, updated_at = NOW()
        WHERE index_jobs.status IN ('done', 'failed')
        """,
        {"t": track_id, "v": TEXT_MODEL_VERSION},
    )


async def process_audio(track_id: int) -> None:
    """Embed the DSP row of one track (see services/rec/audio_embed.py)."""
    backend = get_audio_backend()
    features = await db.fetch_one(
        "SELECT * FROM track_features WHERE track_id = :t", {"t": track_id}
    )
    vec = await backend.embed(track_id, features, await _file_path(track_id))
    if vec is None:
        raise FeaturesNotReady(f"no DSP features for track {track_id}")
    await store_audio(track_id, vec)


async def process(job: dict) -> None:
    track_id = int(job["track_id"])
    if job.get("kind") == "audio":
        await process_audio(track_id)
        return
    if job.get("kind") == "text":
        text = await _text_input(track_id)
        if not text:
            raise ValueError(f"no text for track {track_id}")
        vec = await embed_text(text)
        await store_text(track_id, vec)
        return

    path = await _file_path(track_id)
    if not path:
        raise NoLocalSource(f"no local source for track {track_id}")
    if not Path(path).exists():
        raise NoLocalSource(f"missing local file for track {track_id}: {path}")

    features = await _extract_dsp_isolated(path)
    await store(track_id, features)
    # New DSP features invalidate the semantic tags and the audio embedding.
    await enqueue_track(track_id, kinds=("tags", "audio"), force=True)


async def mark(
    job_id: int, status: str, error: str | None = None, *, terminal: bool = False
) -> None:
    await db.execute(
        """
        UPDATE index_jobs
        SET status = :s, last_error = :e,
            attempts = CASE
                WHEN :s <> 'failed' THEN attempts
                WHEN :terminal THEN 5
                ELSE attempts + 1
            END,
            updated_at = NOW()
        WHERE id = :i
        """,
        {"s": status, "e": error, "i": job_id, "terminal": terminal},
    )


async def drain(limit: int = 0, kinds: tuple[str, ...] = ("dsp", "text")) -> int:
    processed = 0
    dsp_done = 0

    async def _one(job: dict) -> tuple[bool, str | None]:
        kind = job.get("kind")
        try:
            await process(job)
            await mark(job["id"], "done")
            log.info("indexed %s track %s", kind, job["track_id"])
            return True, kind
        except NoLocalSource as exc:
            log.info("index job %s dropped: %s", job["id"], exc)
            await db.execute("DELETE FROM index_jobs WHERE id = :i", {"i": job["id"]})
        except CorruptAudio as exc:
            log.error("index job %s: corrupt audio: %s", job["id"], exc)
            await mark(job["id"], "failed", f"corrupt: {exc}"[:500], terminal=True)
        except DspWorkerCrash as exc:
            log.error("index job %s: DSP worker crash: %s", job["id"], exc)
            await mark(job["id"], "failed", f"dsp_crash: {exc}"[:500], terminal=True)
        except Exception as exc:  # noqa: BLE001
            log.exception("index job %s failed", job["id"])
            await mark(job["id"], "failed", str(exc)[:500])
        return False, kind

    batch_width = _DSP_WORKERS if "dsp" in kinds else 1
    while True:
        remaining = (limit - processed) if limit else batch_width
        take = min(batch_width, remaining) if limit else batch_width
        if take <= 0:
            if dsp_done > 0:
                await rec_features.recompute_corpus_stats()
            return processed
        jobs: list[dict] = []
        for _ in range(take):
            job = await claim(kinds=kinds)
            if not job:
                break
            jobs.append(job)
        if not jobs:
            if dsp_done > 0:
                await rec_features.recompute_corpus_stats()
            return processed
        results = await asyncio.gather(*(_one(j) for j in jobs))
        for ok, kind in results:
            if ok:
                processed += 1
                if kind == "dsp":
                    dsp_done += 1
                    if dsp_done % 200 == 0:
                        await rec_features.recompute_corpus_stats()
        if limit and processed >= limit:
            if dsp_done > 0:
                await rec_features.recompute_corpus_stats()
            return processed


async def drain_tags(limit: int = 0, batch_size: int = 12) -> int:
    """Drain `tags` jobs in batches via LLM + heuristic fallback."""
    processed = 0
    while True:
        size = min(batch_size, limit - processed) if limit else batch_size
        if size <= 0:
            return processed
        jobs = await claim_tags_batch(size)
        if not jobs:
            if processed > 0:
                await rec_features.recompute_corpus_stats()
            return processed
        track_ids = [int(j["track_id"]) for j in jobs]
        rows = await db.fetch_all(
            """
            SELECT t.id AS track_id, t.title, t.artists, t.album, t.genre, t.genres_norm, t.year,
                   tf.bpm, tf.energy, tf.valence, tf.danceability, tf.dsp_extra, tf.lyrics
            FROM tracks t
            LEFT JOIN track_features tf ON tf.track_id = t.id
            WHERE t.id = ANY(:ids)
            """,
            {"ids": track_ids},
        )
        by_tid = {int(r["track_id"]): r for r in rows}
        ordered_rows = [by_tid[tid] for tid in track_ids if tid in by_tid]
        try:
            tagged_map = await rec_tagger.tag_batch(ordered_rows)
            for job in jobs:
                tid = int(job["track_id"])
                tagged = tagged_map.get(tid)
                if tagged is None:
                    await db.execute("DELETE FROM index_jobs WHERE id = :i", {"i": job["id"]})
                    continue
                await store_tags(tid, tagged)
                await _refresh_text_job(tid)
                await mark(job["id"], "done")
                processed += 1
        except Exception as exc:  # noqa: BLE001
            log.exception("tags batch failed")
            for job in jobs:
                await mark(job["id"], "failed", str(exc)[:500])


async def _process_audio_job(job: dict) -> bool:
    try:
        await process(job)
        await mark(job["id"], "done")
        return True
    except FeaturesNotReady as exc:
        log.info("audio job %s deferred: %s", job["id"], exc)
        await mark(job["id"], "failed", str(exc)[:500])
    except Exception as exc:  # noqa: BLE001
        log.exception("audio job %s failed", job["id"])
        await mark(job["id"], "failed", str(exc)[:500])
    return False


async def _run_audio_job(job: dict, worker_slots: asyncio.Semaphore) -> bool:
    async with worker_slots:
        return await _process_audio_job(job)


async def drain_audio(limit: int = 0) -> int:
    await rec_features.load_corpus_stats()
    batch = max(1, settings.indexer_audio_batch)
    workers = max(1, settings.indexer_audio_workers)
    processed = 0
    while True:
        size = min(batch, limit - processed) if limit else batch
        if size <= 0:
            return processed
        jobs = await claim_audio_batch(size)
        if not jobs:
            return processed
        worker_slots = asyncio.Semaphore(workers)
        done = await asyncio.gather(*(_run_audio_job(job, worker_slots) for job in jobs))
        processed += sum(done)


async def _enqueue_all() -> tuple[int, int, int, int]:
    return (
        await enqueue_local(),
        await enqueue_tags(),
        await enqueue_text(),
        await enqueue_audio(),
    )


async def reset_stale_running(max_age_minutes: int = 2) -> int:
    rows = await db.execute_returning(
        """
        UPDATE index_jobs
        SET status = 'pending', updated_at = NOW()
        WHERE kind IN ('dsp', 'tags', 'audio', 'text') AND status = 'running'
          AND updated_at < NOW() - make_interval(mins => :m)
        RETURNING id
        """,
        {"m": max_age_minutes},
    )
    return len(rows)


async def requeue_failed() -> int:
    rows = await db.execute_returning(
        """
        UPDATE index_jobs
        SET status = 'pending', attempts = 0, last_error = NULL, updated_at = NOW()
        WHERE kind = 'dsp' AND status = 'failed'
        RETURNING id
        """
    )
    return len(rows)


async def requeue_audio_all() -> int:
    """Force every audio embedding to be rebuilt with the current corpus stats."""
    rows = await db.execute_returning(
        """
        UPDATE index_jobs
        SET status = 'pending', attempts = 0, last_error = NULL, updated_at = NOW()
        WHERE kind = 'audio' AND model_version = :v
        RETURNING id
        """,
        {"v": AUDIO_MODEL_VERSION},
    )
    return len(rows)


async def requeue_tags_all() -> int:
    """Force re-tagging of every local track (used after a DSP backfill)."""
    rows = await db.execute_returning(
        """
        UPDATE index_jobs
        SET status = 'pending', attempts = 0, last_error = NULL, updated_at = NOW()
        WHERE kind = 'tags' AND model_version = :v
        RETURNING id
        """,
        {"v": TAGS_MODEL_VERSION},
    )
    return len(rows)


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true", help="drain the queue and exit")
    parser.add_argument("--enqueue", action="store_true", help="enqueue local tracks before draining")
    parser.add_argument(
        "--requeue-failed",
        action="store_true",
        help="reset failed DSP jobs",
    )
    parser.add_argument(
        "--audio-only",
        action="store_true",
        help="process only audio-embedding jobs",
    )
    parser.add_argument(
        "--tags-only",
        action="store_true",
        help="process only semantic tagging jobs",
    )
    parser.add_argument(
        "--reembed-audio",
        action="store_true",
        help="recompute corpus stats and re-embed all audio vectors (after DSP backfill)",
    )
    parser.add_argument(
        "--retag-all",
        action="store_true",
        help="force re-tagging of every local track (after DSP backfill)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    if not shutil.which("ffmpeg"):
        log.warning("ffmpeg not found: m4a/opus DSP jobs will fail without the fallback")
    if args.requeue_failed:
        count = await requeue_failed()
        log.info("requeued %s failed dsp jobs", count)
    stale = await reset_stale_running()
    if stale:
        log.info("reset %s stale running jobs", stale)
    if args.enqueue:
        dsp_count, tags_count, text_count, audio_count = await _enqueue_all()
        log.info(
            "enqueued %s dsp, %s tags, %s text, %s audio jobs",
            dsp_count,
            tags_count,
            text_count,
            audio_count,
        )

    if args.reembed_audio:
        await rec_features.recompute_corpus_stats()
        requeued = await requeue_audio_all()
        log.info("re-embedding %s audio vectors with final corpus stats", requeued)

    if args.retag_all:
        requeued = await requeue_tags_all()
        log.info("re-tagging %s local tracks with current DSP features", requeued)

    if args.once:
        if args.audio_only or args.reembed_audio:
            processed = await drain_audio()
        elif args.tags_only or args.retag_all:
            processed = await drain_tags()
        else:
            processed = (
                await drain(kinds=("dsp",))
                + await drain_tags()
                + await drain_audio()
                + await drain(kinds=("text",))
            )
        log.info("processed %s jobs", processed)
        return

    last_enqueue = 0.0
    while True:
        now = time.monotonic()
        if now - last_enqueue >= ENQUEUE_INTERVAL:
            try:
                dsp, tags, text, audio = await _enqueue_all()
                if dsp or tags or text or audio:
                    log.info(
                        "auto-enqueue: %s dsp, %s tags, %s text, %s audio",
                        dsp,
                        tags,
                        text,
                        audio,
                    )
            except Exception:
                log.exception("auto-enqueue failed")
            last_enqueue = now
        if args.audio_only:
            processed = await drain_audio()
        elif args.tags_only:
            processed = await drain_tags()
        else:
            # Progressively advance full pipeline per batch so tags/audio/text land continuously
            processed = (
                await drain(kinds=("dsp",), limit=200)
                + await drain_tags(limit=200)
                + await drain_audio(limit=400)
                + await drain(kinds=("text",), limit=200)
            )
        log.info("idle, processed %s jobs", processed)
        if processed == 0:
            await asyncio.sleep(POLL_SECONDS)


if __name__ == "__main__":
    asyncio.run(main())
