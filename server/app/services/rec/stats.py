"""Nightly and on-demand refresh of recommendation aggregates and stats."""

from __future__ import annotations

import logging
from typing import Any

from ... import db
from . import features as rec_features

log = logging.getLogger("kwhale.rec")

_VIEWS = ("mv_user_track_stats", "mv_user_artist_stats", "mv_item_cooccurrence")


def _preset_for_local_hour(hour: int) -> str:
    if 6 <= hour <= 11:
        return "morning"
    if 12 <= hour <= 16:
        return "focus"
    if 17 <= hour <= 21:
        return "evening_chill"
    return "night"


async def refresh_mood_profiles() -> int:
    """Learn per-user mood centroids from mood-mix events and time-of-day listens."""
    stats = await rec_features.load_corpus_stats()
    rows = await db.fetch_all(
        """
        SELECT e.user_id,
               e.type,
               e.payload,
               COALESCE(e.occurred_at, e.received_at) AS ts,
               tf.loudness_lufs, tf.dynamic_complexity, tf.bpm, tf.beat_confidence,
               tf.danceability, tf.onset_rate, tf.spectral_centroid, tf.spectral_rolloff,
               tf.spectral_flux, tf.energy, tf.valence, tf.mfcc_mean, tf.dsp_extra
        FROM events e
        JOIN track_features tf ON tf.track_id = e.track_id
        WHERE e.track_id IS NOT NULL
          AND e.type IN ('play', 'complete', 'repeat', 'like', 'skip', 'dislike')
          AND COALESCE(e.occurred_at, e.received_at) > NOW() - INTERVAL '180 days'
        ORDER BY ts DESC
        LIMIT 20000
        """
    )

    buckets: dict[tuple[int, str], dict[str, Any]] = {}
    for row in rows:
        vec = rec_features.feature_vector(row, stats)
        if vec is None:
            continue
        user_id = int(row["user_id"])
        payload = row.get("payload") or {}
        if isinstance(payload, str):
            import json

            try:
                payload = json.loads(payload)
            except ValueError:
                payload = {}

        mix_id = ((payload.get("context") or {}).get("mixId") or "").strip()
        preset: str | None = None
        explicit_mix = False
        if mix_id.startswith("mood:"):
            candidate_preset = mix_id.split(":", 1)[1].strip()
            if candidate_preset in ("morning", "focus", "evening_chill", "party", "night"):
                preset = candidate_preset
                explicit_mix = True

        if preset is None:
            tz_min = payload.get("timezoneOffsetMinutes")
            try:
                tz = int(tz_min) if tz_min is not None else 180
            except (TypeError, ValueError):
                tz = 180
            ts = row["ts"]
            if ts is None:
                continue
            local_hour = (ts.hour + (tz // 60)) % 24
            preset = _preset_for_local_hour(local_hour)

        ev_type = row["type"]
        listened = payload.get("listenedMs")
        dur = payload.get("trackDurationMs")
        is_implicit_complete = (
            ev_type in ("skip", "pause")
            and isinstance(listened, (int, float))
            and isinstance(dur, (int, float))
            and dur > 0
            and listened >= 0.85 * dur
        )

        key = (user_id, preset)
        entry = buckets.setdefault(
            key,
            {"pos_sum": [0.0] * len(vec), "pos_w": 0.0, "pos_n": 0, "neg_n": 0},
        )
        base_w = 1.5 if explicit_mix else 0.5
        if ev_type in ("complete", "like", "repeat") or is_implicit_complete:
            w = base_w * (1.4 if ev_type in ("complete", "like") else 1.0)
            for idx, v in enumerate(vec):
                entry["pos_sum"][idx] += w * v
            entry["pos_w"] += w
            entry["pos_n"] += 1
        elif ev_type == "play":
            w = base_w * 0.4
            for idx, v in enumerate(vec):
                entry["pos_sum"][idx] += w * v
            entry["pos_w"] += w
        elif ev_type in ("skip", "dislike"):
            entry["neg_n"] += 1
            if explicit_mix:
                penalty_w = 0.25
                for idx, v in enumerate(vec):
                    entry["pos_sum"][idx] -= penalty_w * v

    updated = 0
    for (user_id, preset), entry in buckets.items():
        if entry["pos_w"] <= 0.0 or entry["pos_n"] < 1:
            continue
        centroid = [round(v / entry["pos_w"], 6) for v in entry["pos_sum"]]
        await db.execute(
            """
            INSERT INTO user_mood_profiles (user_id, preset, audio_centroid, pos_count, neg_count, updated_at)
            VALUES (:u, :p, :c, :pn, :nn, NOW())
            ON CONFLICT (user_id, preset) DO UPDATE SET
                audio_centroid = EXCLUDED.audio_centroid,
                pos_count = EXCLUDED.pos_count,
                neg_count = EXCLUDED.neg_count,
                updated_at = NOW()
            """,
            {
                "u": user_id,
                "p": preset,
                "c": centroid,
                "pn": entry["pos_n"],
                "nn": entry["neg_n"],
            },
        )
        updated += 1
    return updated


async def refresh_stats() -> None:
    for view in _VIEWS:
        try:
            await db.execute(f"REFRESH MATERIALIZED VIEW CONCURRENTLY {view}")
        except Exception:
            try:
                await db.execute(f"REFRESH MATERIALIZED VIEW {view}")
            except Exception:
                log.exception("refresh %s failed", view)
    try:
        await rec_features.recompute_corpus_stats()
    except Exception:
        log.exception("recompute_corpus_stats failed")
    try:
        await refresh_mood_profiles()
    except Exception:
        log.exception("refresh_mood_profiles failed")
