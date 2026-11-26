"""User taste profile from materialized views, favorites, onboarding, and recent events."""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta
from typing import Any

from ... import db
from ..genres import genre_families
from .features import feature_vector, load_corpus_stats


def _days_since(dt: datetime | None, now: datetime) -> float:
    if dt is None:
        return 30.0
    ts = dt if dt.tzinfo else dt.replace(tzinfo=UTC)
    return max(0.0, (now - ts).total_seconds() / 86400.0)


def _saturate(raw_by_key: dict[str, float], half_point: float = 6.0) -> dict[str, float]:
    """Smooth Bayesian-style saturation into (0, 1] without single-play spikes to 1.0."""
    positives = {k: v for k, v in raw_by_key.items() if v > 0.0}
    if not positives:
        return {}
    peak = max(max(positives.values()), half_point)
    out: dict[str, float] = {}
    for key, value in positives.items():
        # Blend absolute confidence (x / (x + k)) with relative rank (x / peak)
        abs_conf = value / (value + half_point)
        rel_conf = value / peak
        out[key] = min(1.0, round(0.55 * rel_conf + 0.45 * abs_conf, 4))
    return out


async def _track_interactions(user_id: int) -> list[dict[str, Any]]:
    """Combine play/complete/skip stats with favorites (including unplayed likes)."""
    return await db.fetch_all(
        """
        SELECT COALESCE(s.track_id, f.track_id) AS track_id,
               t.artists,
               t.genre,
               t.genres_norm,
               COALESCE(s.plays, 0)     AS plays,
               COALESCE(s.completes, 0) AS completes,
               COALESCE(s.skips, 0)     AS skips,
               COALESCE(s.repeats, 0)   AS repeats,
               s.last_played,
               (f.track_id IS NOT NULL) AS liked,
               f.created_at             AS liked_at
        FROM mv_user_track_stats s
        FULL OUTER JOIN favorites f
          ON f.user_id = s.user_id AND f.track_id = s.track_id
        JOIN tracks t ON t.id = COALESCE(s.track_id, f.track_id)
        WHERE COALESCE(s.user_id, f.user_id) = :u
        """,
        {"u": user_id},
    )


async def _onboarding(user_id: int) -> dict[str, Any]:
    row = await db.fetch_one(
        "SELECT genres, artists, artist_ids FROM user_onboarding WHERE user_id = :u",
        {"u": user_id},
    )
    if not row:
        return {"genres": [], "artists": []}
    names: list[str] = [str(a).strip().lower() for a in (row.get("artists") or []) if str(a).strip()]
    raw_ids = [int(aid) for aid in (row.get("artist_ids") or []) if str(aid).isdigit()]
    if raw_ids:
        ar_rows = await db.fetch_all(
            "SELECT name FROM artists WHERE id = ANY(:ids)", {"ids": raw_ids}
        )
        for ar in ar_rows:
            nm = (ar.get("name") or "").strip().lower()
            if nm and nm not in names:
                names.append(nm)
    return {"genres": list(row.get("genres") or []), "artists": names}


async def load_artist_affinity(
    user_id: int,
    interactions: list[dict[str, Any]] | None = None,
    onboarding: dict[str, Any] | None = None,
) -> dict[str, float]:
    rows = interactions if interactions is not None else await _track_interactions(user_id)
    now = datetime.now(UTC)
    by_artist: dict[str, list[float]] = {}

    for row in rows:
        artists = [str(a).strip().lower() for a in (row.get("artists") or []) if str(a).strip()]
        if not artists:
            continue
        liked = bool(row.get("liked"))
        plays = float(row.get("plays") or 0)
        completes = float(row.get("completes") or 0)
        repeats = float(row.get("repeats") or 0)
        skips = float(row.get("skips") or 0)

        days = _days_since(row.get("last_played") or row.get("liked_at"), now)
        decay = math.exp(-days / 90.0)
        like_decay = math.exp(-_days_since(row.get("liked_at"), now) / 365.0) if liked else 0.0

        pos = 3.0 * like_decay + (1.8 * completes + 1.0 * repeats + 0.35 * plays) * decay
        neg = 0.7 * skips * decay
        net = max(-1.5, pos - neg)
        if net == 0.0:
            continue

        for idx, name in enumerate(artists):
            # Primary artist gets full credit; featured collaborators get 0.5x.
            factor = 1.0 if idx == 0 else 0.5
            by_artist.setdefault(name, []).append(net * factor)

    raw: dict[str, float] = {}
    for name, scores in by_artist.items():
        # Diminishing returns per extra track so 120 passive tracks cannot swamp 8 loved ones.
        scores.sort(reverse=True)
        agg = sum(val / (1.0 + 0.18 * idx) for idx, val in enumerate(scores[:25]))
        if agg > 0.0:
            raw[name] = agg

    # Cold-start onboarding prior when explicit/listening history is still small.
    if len(raw) < 5:
        ob = onboarding if onboarding is not None else await _onboarding(user_id)
        for name in ob.get("artists") or []:
            raw.setdefault(name, 2.5)

    return _saturate(raw, half_point=5.0)


async def load_genre_affinity(
    user_id: int,
    interactions: list[dict[str, Any]] | None = None,
    onboarding: dict[str, Any] | None = None,
) -> dict[str, float]:
    """Affinity per canonical genre family (including liked tracks without plays)."""
    rows = interactions if interactions is not None else await _track_interactions(user_id)
    now = datetime.now(UTC)
    by_family: dict[str, list[float]] = {}

    for row in rows:
        families = list(row.get("genres_norm") or [])
        if not families:
            families = genre_families(row.get("genre"))
        if not families:
            continue
        liked = bool(row.get("liked"))
        plays = float(row.get("plays") or 0)
        completes = float(row.get("completes") or 0)
        repeats = float(row.get("repeats") or 0)
        skips = float(row.get("skips") or 0)

        days = _days_since(row.get("last_played") or row.get("liked_at"), now)
        decay = math.exp(-days / 90.0)
        like_decay = math.exp(-_days_since(row.get("liked_at"), now) / 365.0) if liked else 0.0

        weight = 3.0 * like_decay + (1.8 * completes + 1.0 * repeats + 0.35 * plays) * decay - 0.5 * skips * decay
        if weight <= 0.0:
            continue
        for family in families:
            by_family.setdefault(family, []).append(weight)

    raw: dict[str, float] = {}
    for family, scores in by_family.items():
        scores.sort(reverse=True)
        raw[family] = sum(val / (1.0 + 0.08 * idx) for idx, val in enumerate(scores[:40]))

    if len(raw) < 3:
        ob = onboarding if onboarding is not None else await _onboarding(user_id)
        for raw_g in ob.get("genres") or []:
            for fam in genre_families(raw_g):
                raw.setdefault(fam, 3.0)

    return _saturate(raw, half_point=8.0)


async def load_context(user_id: int) -> dict:
    """Blend the user's local hour-of-day profile with the mood of recent non-skipped plays."""
    tz_row = await db.fetch_one(
        """
        SELECT (payload->>'timezoneOffsetMinutes')::int AS tz
        FROM events
        WHERE user_id = :u
          AND (payload->>'timezoneOffsetMinutes') ~ '^-?\\d+$'
        ORDER BY COALESCE(occurred_at, received_at) DESC
        LIMIT 1
        """,
        {"u": user_id},
    )
    tz_offset = int(tz_row["tz"]) if tz_row and tz_row.get("tz") is not None else 180
    local_hour = (datetime.now(UTC) + timedelta(minutes=tz_offset)).hour

    hourly = await db.fetch_one(
        """
        SELECT AVG(tf.energy) AS energy, AVG(tf.valence) AS valence, AVG(tf.bpm) AS bpm
        FROM events e
        JOIN track_features tf ON tf.track_id = e.track_id
        WHERE e.user_id = :u
          AND e.type IN ('play', 'complete', 'repeat')
          AND tf.energy IS NOT NULL
          AND EXTRACT(
              HOUR FROM (
                  COALESCE(e.occurred_at, e.received_at)
                  + make_interval(mins => COALESCE(
                      CASE WHEN (e.payload->>'timezoneOffsetMinutes') ~ '^-?\\d+$'
                           THEN (e.payload->>'timezoneOffsetMinutes')::int
                           ELSE :tz END,
                      :tz
                  ))
              )
          ) = :h
        """,
        {"u": user_id, "h": local_hour, "tz": tz_offset},
    )
    recent = await db.fetch_all(
        """
        SELECT tf.energy, tf.valence, tf.bpm
        FROM events e
        JOIN track_features tf ON tf.track_id = e.track_id
        WHERE e.user_id = :u
          AND e.type IN ('play', 'complete', 'repeat')
          AND tf.energy IS NOT NULL
          AND NOT EXISTS (
              SELECT 1 FROM events sk
              WHERE sk.user_id = e.user_id
                AND sk.track_id = e.track_id
                AND sk.playback_id IS NOT NULL
                AND sk.playback_id = e.playback_id
                AND sk.type = 'skip'
                AND (sk.payload->>'listenedMs') ~ '^\\d+$'
                AND (sk.payload->>'listenedMs')::bigint < 25000
          )
        ORDER BY COALESCE(e.occurred_at, e.received_at) DESC
        LIMIT 3
        """,
        {"u": user_id},
    )

    def _blend_field(field: str) -> float | None:
        parts: list[tuple[float, float]] = []
        h_val = (hourly or {}).get(field)
        if h_val is not None:
            parts.append((0.55, float(h_val)))
        r_vals = [float(r[field]) for r in recent if r.get(field) is not None]
        if r_vals:
            parts.append((0.45, sum(r_vals) / len(r_vals)))
        total = sum(w for w, _ in parts)
        return (sum(w * v for w, v in parts) / total) if total > 0 else None

    return {
        "energy": _blend_field("energy"),
        "valence": _blend_field("valence"),
        "bpm": _blend_field("bpm"),
        "local_hour": local_hour,
    }


async def load_seed_ids(user_id: int, limit: int = 50) -> list[int]:
    rows = await db.fetch_all(
        """
        SELECT track_id, MAX(score) AS score
        FROM (
            SELECT track_id, 3.0 AS score
            FROM favorites
            WHERE user_id = :u
            UNION ALL
            SELECT track_id, (2.0 * completes + 1.5 * repeats + 0.3 * plays - skips) AS score
            FROM mv_user_track_stats
            WHERE user_id = :u AND (completes >= 1 OR repeats >= 1 OR plays >= 2)
        ) q
        GROUP BY track_id
        HAVING MAX(score) > 0
        ORDER BY MAX(score) DESC, track_id DESC
        LIMIT :l
        """,
        {"u": user_id, "l": limit},
    )
    return [int(r["track_id"]) for r in rows]


def _weighted_mean(parts: list[tuple[float, list[float]]]) -> list[float] | None:
    total = sum(weight for weight, _ in parts)
    if total <= 0.0 or not parts:
        return None
    dims = len(parts[0][1])
    out = [0.0] * dims
    for weight, vec in parts:
        if len(vec) != dims:
            continue
        for index, value in enumerate(vec):
            out[index] += weight * value
    norm = math.sqrt(sum(v * v for v in out))
    if norm <= 1e-9:
        return [value / total for value in out]
    return [round(v / norm, 6) for v in out]


async def load_taste_vectors(user_id: int) -> dict[str, Any]:
    """Weighted Z-scored DSP and text centroids of what the user actually likes/completes."""
    stats = await load_corpus_stats()
    rows = await db.fetch_all(
        """
        SELECT tf.*,
               COALESCE(st.plays, 0)     AS plays,
               COALESCE(st.completes, 0) AS completes,
               COALESCE(st.skips, 0)     AS skips,
               COALESCE(st.repeats, 0)   AS repeats,
               st.last_played,
               (f.track_id IS NOT NULL)  AS liked,
               f.created_at              AS liked_at,
               te.vec::text              AS text_vec
        FROM track_features tf
        LEFT JOIN mv_user_track_stats st
          ON st.track_id = tf.track_id AND st.user_id = :u
        LEFT JOIN favorites f
          ON f.track_id = tf.track_id AND f.user_id = :u
        LEFT JOIN track_emb_text te
          ON te.track_id = tf.track_id
        WHERE st.track_id IS NOT NULL OR f.track_id IS NOT NULL
        """,
        {"u": user_id},
    )
    now = datetime.now(UTC)
    taste_parts: list[tuple[float, list[float]]] = []
    text_parts: list[tuple[float, list[float]]] = []
    skip_parts: list[tuple[float, list[float]]] = []
    recent_rows: list[tuple[datetime, list[float]]] = []

    for row in rows:
        liked = bool(row.get("liked"))
        plays = float(row.get("plays") or 0)
        completes = float(row.get("completes") or 0)
        repeats = float(row.get("repeats") or 0)
        skips = float(row.get("skips") or 0)
        ts = row.get("last_played") or row.get("liked_at")
        days = _days_since(ts, now)
        decay = math.exp(-days / 120.0)
        weight = (
            (3.0 if liked else 0.0)
            + (1.8 * completes + 1.0 * repeats + 0.35 * plays) * decay
            - 0.6 * skips * decay
        )

        vec = feature_vector(row, stats)
        if vec is not None:
            if weight > 0.0:
                taste_parts.append((weight, vec))
                if ts is not None:
                    ts_tz = ts if ts.tzinfo else ts.replace(tzinfo=UTC)
                    recent_rows.append((ts_tz, vec))
            if skips >= 2 and completes == 0 and not liked:
                skip_parts.append((skips, vec))

        if (liked or completes > 0 or plays >= 2) and weight > 0.5 and len(text_parts) < 100 and row.get("text_vec"):
            try:
                tvec = [float(x) for x in str(row["text_vec"]).strip("[]").split(",")]
                if tvec:
                    text_parts.append((weight, tvec))
            except ValueError:
                pass

    recent_rows.sort(key=lambda pair: pair[0], reverse=True)
    recent_parts = [(1.0 / (1.0 + 0.05 * idx), vec) for idx, (_, vec) in enumerate(recent_rows[:40])]

    mood_rows = await db.fetch_all(
        "SELECT preset, audio_centroid FROM user_mood_profiles WHERE user_id = :u AND pos_count >= 2",
        {"u": user_id},
    )
    mood_centroids = {
        str(r["preset"]): [float(x) for x in r["audio_centroid"]]
        for r in mood_rows
        if r.get("audio_centroid")
    }

    return {
        "taste": _weighted_mean(taste_parts),
        "recent": _weighted_mean(recent_parts),
        "skip": _weighted_mean(skip_parts),
        "text_taste": _weighted_mean(text_parts),
        "moods": mood_centroids,
    }


def _history_span_days(interactions: list[dict[str, Any]]) -> float:
    if not interactions:
        return 60.0
    now = datetime.now(UTC)
    max_days = 0.0
    for row in interactions:
        for ts in (row.get("last_played"), row.get("liked_at")):
            if ts is not None:
                max_days = max(max_days, _days_since(ts, now))
    return max_days if max_days > 0.0 else 60.0


async def load_profile(user_id: int) -> dict[str, Any]:
    interactions = await _track_interactions(user_id)
    onboarding = await _onboarding(user_id)
    return {
        "artists": await load_artist_affinity(user_id, interactions, onboarding),
        "genres": await load_genre_affinity(user_id, interactions, onboarding),
        "context": await load_context(user_id),
        "seeds": await load_seed_ids(user_id),
        "taste": await load_taste_vectors(user_id),
        "history_days": _history_span_days(interactions),
    }
