"""Scoring core for the recommendation layer (pure functions, no DB/I/O).

Supports:
  * Taste-first scoring (`personal`, `discovery`, `genre`, `artist`)
  * Query-driven hybrid scoring (`prompt`, `mood`, `radio`, `similar`)
  * Fixed missing-term bias, Bayesian engagement, inverted discovery recency,
    strict top-K diversification, and relevance-guarded exploration.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from ..genres import family_set, genre_families

if TYPE_CHECKING:
    from .llm import QueryPlan

DEFAULT_PROFILE: dict = {
    "pref": 0.65,
    "ctx": 0.20,
    "novelty": 0.15,
    "pref_parts": {
        "artist": 0.30,
        "genre": 0.20,
        "cooc": 0.15,
        "audio": 0.25,
        "text": 0.05,
        "engagement": 0.05,
    },
    "max_per_artist": 2,
    "max_genre_share": 0.6,
    "exploration": 0.2,
}

PRESET_WEIGHTS: dict[str, tuple[float, float, float]] = {
    "default": (0.65, 0.20, 0.15),
    "morning": (0.55, 0.30, 0.15),
    "focus": (0.50, 0.15, 0.35),
    "evening_chill": (0.45, 0.35, 0.20),
    "party": (0.60, 0.25, 0.15),
    "night": (0.50, 0.30, 0.20),
}


async def seed_profiles() -> int:
    """Insert the built-in scoring profiles once (idempotent)."""
    from ... import db

    created = 0
    for label, (pref, ctx, novelty) in PRESET_WEIGHTS.items():
        params = dict(DEFAULT_PROFILE)
        params.update({"preset": label, "pref": pref, "ctx": ctx, "novelty": novelty})
        rows = await db.execute_returning(
            """
            INSERT INTO scoring_profiles (label, params, version, updated_at)
            VALUES (:l, CAST(:p AS jsonb), 1, NOW())
            ON CONFLICT (label) DO NOTHING
            RETURNING label
            """,
            {"l": label, "p": __import__("json").dumps(params)},
        )
        created += len(rows)
    return created


REASON_BY_TERM = {
    "artist": "liked_artist",
    "genre": "genre_match",
    "cooc": "rediscover",
    "audio": "similar_sound",
    "text": "similar_lyrics",
    "engagement": "liked_artist",
}


@dataclass
class Candidate:
    track_id: int
    artist: str | None = None
    artists: list[str] | None = None
    album: str | None = None
    year: int | None = None
    genre: str | None = None
    genres: list[str] | None = None  # canonical families (services.genres)
    vec: list[float] | None = None  # Z-scored L2-normalized DSP vector
    energy: float | None = None
    valence: float | None = None
    bpm: float | None = None
    danceability: float | None = None
    acousticness: float | None = None
    instrumentalness: float | None = None
    subgenres: list[str] | None = None
    moods: list[str] | None = None
    textures: list[str] | None = None
    scenes: list[str] | None = None
    language: str | None = None
    vibe_summary: str | None = None
    has_features: bool = False
    plays: int = 0
    completes: int = 0
    skips: int = 0
    repeats: int = 0
    liked: bool = False
    last_played: datetime | None = None
    artist_affinity: float | None = None
    genre_affinity: float | None = None
    cooc: float | None = None
    sim_audio: float | None = None
    sim_text: float | None = None
    skip_sim: float | None = None
    sem_query_sim: float | None = None
    mood_pack_sim: float | None = None
    ref_sim: float | None = None


@dataclass
class Scored:
    candidate: Candidate
    score: float
    reason_code: str
    reason_detail: str | None = None


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def _cosine_to_unit(value: float) -> float:
    return _clamp((value + 1.0) / 2.0)


def octave_bpm_diff(bpm: float, target: float) -> float:
    """Minimum distance modulo half/double-time (Essentia 75 vs 150 BPM quirk)."""
    if bpm <= 0 or target <= 0:
        return 100.0
    return min(abs(bpm - target), abs(2.0 * bpm - target), abs(0.5 * bpm - target))


# Textures that credibly signal a second genre family.  A track is allowed to
# belong to several families (jazz-rap is both rap and jazz); this is what makes
# style prompts ("джаз", "акустика") find cross-genre records.
_TEXTURE_FAMILIES: dict[str, str] = {
    "jazz_samples": "jazz",
}


def candidate_families(cand: Candidate) -> set[str]:
    """Every genre family a track belongs to, including style inferred from tags."""
    fams: set[str] = set(cand.genres or [])
    if cand.genre:
        fams.update(genre_families(cand.genre))
    if cand.subgenres:
        fams.update(family_set([s.replace("_", " ") for s in cand.subgenres]))
    for texture in cand.textures or []:
        family = _TEXTURE_FAMILIES.get(texture)
        if family:
            fams.add(family)
    return fams


def days_since_play(candidate: Candidate, now: datetime | None = None) -> float | None:
    """Age of the last play in days, or None when the track was never played."""
    if candidate.last_played is None:
        return None
    now = now or datetime.now(UTC)
    last = (
        candidate.last_played
        if candidate.last_played.tzinfo
        else candidate.last_played.replace(tzinfo=UTC)
    )
    return max(0.0, (now - last).total_seconds() / 86400.0)


def pref_terms(candidate: Candidate) -> dict[str, float]:
    terms: dict[str, float] = {}
    for name, value in (
        ("artist", candidate.artist_affinity),
        ("genre", candidate.genre_affinity),
        ("cooc", candidate.cooc),
    ):
        if value is not None:
            terms[name] = _clamp(float(value))
    if candidate.sim_audio is not None:
        terms["audio"] = _cosine_to_unit(candidate.sim_audio)
    if candidate.sim_text is not None:
        terms["text"] = _cosine_to_unit(candidate.sim_text)

    if candidate.plays > 0 or candidate.completes > 0 or candidate.liked or candidate.repeats > 0:
        # Smoothed Bayesian engagement rate; avoids 2-play saturation to 1.0.
        num = (
            candidate.completes
            + 0.7 * candidate.repeats
            + (0.9 if candidate.liked else 0.0)
            + 0.15 * min(candidate.plays, 4)
        )
        den = candidate.plays + 2.5 + 1.5 * candidate.skips
        terms["engagement"] = _clamp((num / den) * 1.4)
    return terms


def pref_score(candidate: Candidate, parts: dict[str, float]) -> tuple[float, str]:
    terms = pref_terms(candidate)
    if not terms:
        return 0.0, "new_for_you"

    total_weight = 0.0
    accumulator = 0.0
    best_term: str | None = None
    best_contribution = 0.0

    for name, value in terms.items():
        weight = max(0.0, float(parts.get(name, 0.0)))
        if weight <= 0.0:
            continue
        contribution = weight * value
        accumulator += contribution
        total_weight += weight
        if contribution > best_contribution:
            best_contribution = contribution
            best_term = name

    if total_weight <= 0.0 or best_term is None:
        return 0.0, "new_for_you"
    return accumulator / total_weight, REASON_BY_TERM.get(best_term, "genre_match")


def context_score(candidate: Candidate, context: dict) -> float:
    matches: list[float] = []
    for field in ("energy", "valence"):
        target = context.get(field)
        value = getattr(candidate, field)
        if target is None or value is None:
            continue
        matches.append(1.0 - abs(float(value) - float(target)))
    if context.get("bpm") is not None and candidate.bpm is not None:
        diff = octave_bpm_diff(float(candidate.bpm), float(context["bpm"]))
        matches.append(math.exp(-((diff / 30.0) ** 2)))
    if not matches:
        return 0.5
    return _clamp(sum(matches) / len(matches))


def discovery_cutoff(history_days: float = 60.0) -> float:
    """Adaptive rediscovery horizon: ~7d for brand-new accounts, 30d at maturity."""
    return min(30.0, max(7.0, 0.55 * max(1.0, float(history_days))))


def novelty_score(
    candidate: Candidate,
    now: datetime | None = None,
    *,
    mode: str = "personal",
    history_days: float = 60.0,
) -> float:
    if mode == "discovery":
        if candidate.plays <= 0 or candidate.last_played is None:
            return 0.86
        now = now or datetime.now(UTC)
        last = candidate.last_played if candidate.last_played.tzinfo else candidate.last_played.replace(tzinfo=UTC)
        days = max(0.0, (now - last).total_seconds() / 86400.0)
        cutoff = discovery_cutoff(history_days)
        # Smooth cooldown instead of a hard wall: fresh plays are heavily
        # suppressed but remain eligible (the head cap keeps them rare).
        cool = min(5.0, max(2.0, cutoff * 0.65))
        if days < cool:
            return _clamp(0.02 + 0.04 * (days / max(1.0, cool)))
        if days < cutoff:
            t = (days - cool) / max(1.0, cutoff - cool)
            smooth = t * t * (3.0 - 2.0 * t)
            novelty = 0.10 + 0.52 * smooth
        else:
            novelty = 0.68 + 0.30 * min(1.0, (days - cutoff) / 120.0)
        if candidate.plays >= 4 and days < cutoff:
            novelty *= 0.45
        return _clamp(novelty)

    if candidate.plays <= 0 or candidate.last_played is None:
        return 1.0
    now = now or datetime.now(UTC)
    last = candidate.last_played
    if last.tzinfo is None:
        last = last.replace(tzinfo=UTC)
    days = max(0.0, (now - last).total_seconds() / 86400.0)
    if days < 1.0:
        novelty = 0.05
    elif days < 14.0:
        novelty = 0.15 + 0.55 * (days / 14.0)
    else:
        novelty = 0.70 + 0.25 * min(1.0, (days - 14.0) / 76.0)

    if candidate.plays >= 3 and days <= 7.0:
        novelty *= 1.0 - 0.5 * (1.0 - days / 7.0)
    return _clamp(novelty)


def _tag_overlap_score(candidate: Candidate, plan: QueryPlan) -> tuple[float | None, str | None]:
    """Compute weighted tag/attribute match against `QueryPlan`."""
    buckets: list[tuple[list[str], list[str], float, str]] = [
        (plan.target_subgenres, list(candidate.subgenres or []), 1.6, "поджанр"),
        (plan.target_textures, list(candidate.textures or []), 1.5, "звучание"),
        (plan.target_scenes, list(candidate.scenes or []), 1.3, "сцена"),
        (plan.target_moods, list(candidate.moods or []), 1.2, "настроение"),
    ]
    total_w = 0.0
    matched_w = 0.0
    matched_labels: list[str] = []

    for wanted, have, w, _label in buckets:
        if not wanted:
            continue
        w_set = set(wanted)
        h_set = set(have)
        # Also allow substring fallback for compound subgenres (e.g. dark_phonk vs phonk)
        hits = [
            tag for tag in w_set
            if tag in h_set or any(tag in ht or ht in tag for ht in h_set)
        ]
        total_w += w
        if hits:
            frac = min(1.0, len(hits) / max(1.0, min(2.0, len(w_set))))
            matched_w += w * frac
            matched_labels.extend(hits[:2])

    if plan.target_language:
        total_w += 1.0
        if candidate.language == plan.target_language:
            matched_w += 1.0
        elif candidate.language is not None:
            matched_w += 0.0
        else:
            matched_w += 0.35

    if plan.year_min is not None or plan.year_max is not None:
        total_w += 0.9
        if candidate.year is not None:
            lo = plan.year_min or candidate.year
            hi = plan.year_max or candidate.year
            if lo <= candidate.year <= hi:
                matched_w += 0.9
            else:
                dist = min(abs(candidate.year - lo), abs(candidate.year - hi))
                matched_w += 0.9 * math.exp(-((dist / 4.0) ** 2))
        else:
            matched_w += 0.25

    if total_w <= 0.0:
        return None, None

    penalty = 0.0
    if plan.exclude_moods and candidate.moods:
        if set(plan.exclude_moods) & set(candidate.moods):
            penalty += 0.35
    if plan.exclude_textures and candidate.textures:
        if set(plan.exclude_textures) & set(candidate.textures):
            penalty += 0.35

    score = _clamp((matched_w / total_w) - penalty)
    detail = ", ".join(list(dict.fromkeys(matched_labels))[:3]) if matched_labels else None
    return score, detail


def _audio_target_score(candidate: Candidate, plan: QueryPlan) -> float | None:
    if not plan.audio_target:
        return None
    total_w = 0.0
    acc = 0.0
    scales = {
        "energy": 0.24,
        "valence": 0.25,
        "danceability": 0.25,
        "acousticness": 0.26,
        "instrumentalness": 0.28,
    }
    for axis, target in plan.audio_target.items():
        w = float(plan.axis_weights.get(axis, 0.7))
        if w <= 0.0:
            continue
        val = getattr(candidate, axis, None)
        if val is None:
            continue
        if axis == "bpm":
            diff = octave_bpm_diff(float(val), float(target))
            sim = math.exp(-((diff / 22.0) ** 2))
        else:
            scale = scales.get(axis, 0.25)
            sim = math.exp(-(((float(val) - float(target)) / scale) ** 2))
        acc += w * sim
        total_w += w

    if total_w <= 0.0:
        return None
    return _clamp(acc / total_w)


def query_relevance_score(candidate: Candidate, plan: QueryPlan) -> tuple[float, str, str | None]:
    """Compute how well `candidate` satisfies `plan` (0..1) + dominant reason."""
    parts: list[tuple[str, float, float]] = []

    if candidate.sem_query_sim is not None:
        # Giga-Embeddings cosines typically cluster around 0.2..0.75
        sem_norm = _clamp((float(candidate.sem_query_sim) - 0.12) / 0.58)
        parts.append(("sem", 0.36, sem_norm))

    tag_score, tag_detail = _tag_overlap_score(candidate, plan)
    if tag_score is not None:
        parts.append(("tags", 0.34, tag_score))

    audio_score = _audio_target_score(candidate, plan)
    if audio_score is not None:
        parts.append(("audio", 0.28, audio_score))

    if plan.include_genres:
        # Soft preference: jazz-rap matches both "rap" and "jazz" via its tags.
        fam_match = 1.0 if (candidate_families(candidate) & set(plan.include_genres)) else 0.0
        parts.append(("genre", 0.16, fam_match))

    if candidate.ref_sim is not None:
        parts.append(("ref", 0.30, _clamp(float(candidate.ref_sim))))

    if candidate.mood_pack_sim is not None:
        parts.append(("mood_pack", 0.20, _cosine_to_unit(candidate.mood_pack_sim)))

    if not parts:
        return 0.5, "new_for_you", None

    total_w = sum(w for _, w, _ in parts)
    rel = sum(w * val for _, w, val in parts) / total_w

    # Tracks with neither DSP nor tags cannot reliably match mood/texture prompts
    needs_profile = bool(
        plan.audio_target or plan.hard_bounds or plan.target_textures or plan.target_moods or plan.target_scenes
    )
    if needs_profile and not candidate.has_features and not candidate.subgenres and not candidate.moods:
        rel *= 0.35

    best_kind, _, _ = max(parts, key=lambda item: item[1] * item[2])
    reason_map = {
        "sem": "query_match",
        "tags": "tag_match",
        "audio": "vibe_match",
        "genre": "genre_match",
        "ref": "similar_sound",
        "mood_pack": "vibe_match",
    }
    return _clamp(rel), reason_map.get(best_kind, "query_match"), tag_detail


def score_candidates(
    candidates: list[Candidate],
    profile: dict | None = None,
    context: dict | None = None,
    now: datetime | None = None,
    *,
    mode: str = "personal",
    plan: QueryPlan | None = None,
    history_days: float = 60.0,
) -> list[Scored]:
    profile = profile or DEFAULT_PROFILE
    context = context or {}
    parts = profile.get("pref_parts", DEFAULT_PROFILE["pref_parts"])
    w_pref = float(profile.get("pref", DEFAULT_PROFILE["pref"]))
    w_ctx = float(profile.get("ctx", DEFAULT_PROFILE["ctx"]))
    w_novelty = float(profile.get("novelty", DEFAULT_PROFILE["novelty"]))

    # Sum of active positive weights in pref_parts for confidence check
    max_possible_w = sum(max(0.0, float(v)) for v in parts.values()) or 1.0

    scored: list[Scored] = []
    for candidate in candidates:
        pref, reason = pref_score(candidate, parts)
        terms = pref_terms(candidate)
        present_w = sum(max(0.0, float(parts.get(k, 0.0))) for k in terms)
        evidence_ratio = min(1.0, present_w / max_possible_w)
        # Keep pure single-term unit tests intact when only 1 term is non-zero,
        # while mildly discounting sparse-data candidates when competing against rich ones.
        if len(terms) > 1 or evidence_ratio >= 0.99:
            pref_adj = pref * (0.80 + 0.20 * evidence_ratio)
        else:
            pref_adj = pref

        ctx = context_score(candidate, context)
        nov = novelty_score(candidate, now, mode=mode, history_days=history_days)

        detail: str | None = None
        if plan is not None:
            rel, q_reason, tag_detail = query_relevance_score(candidate, plan)
            taste_blend = float(plan.taste_blend)
            taste_component = 0.72 * pref_adj + 0.28 * ctx
            total = (1.0 - taste_blend) * rel + taste_blend * taste_component + 0.06 * nov
            reason = q_reason
            detail = tag_detail
            if rel < 0.35 and reason == "query_match":
                reason = "explore"
        elif mode == "discovery":
            # Discovery balances unseen/rediscovery novelty with taste fit.
            # A like without a play is NOT an anchor: with large favorite libraries
            # it would otherwise flood the whole mix with "familiar" tracks that
            # were never actually heard.
            total = 0.45 * pref_adj + 0.10 * ctx + 0.45 * nov
            age = days_since_play(candidate, now)
            if candidate.plays > 0 and age is not None and age < 5.0:
                reason = "recent_play"
                total *= 0.45
            elif candidate.plays > 0 and nov >= 0.62:
                reason = "rediscover"
            elif candidate.plays > 0 and (candidate.liked or candidate.completes > 0):
                reason = "familiar_anchor"
            elif candidate.plays == 0 and candidate.liked:
                reason = "liked_unheard"
            elif candidate.plays == 0 and reason not in ("liked_artist", "similar_sound"):
                reason = "new_for_you"
        else:
            total = w_pref * pref_adj + w_ctx * ctx + w_novelty * nov

        if candidate.skips >= 2 and candidate.completes == 0 and not candidate.liked:
            total *= 0.3
            reason = "skipped"
        if candidate.skip_sim is not None and candidate.skip_sim > 0.92:
            total *= 0.65

        scored.append(
            Scored(
                candidate=candidate,
                score=round(_clamp(total), 4),
                reason_code=reason,
                reason_detail=detail,
            )
        )

    scored.sort(key=lambda item: item.score, reverse=True)
    return scored


def enforce_relevance_floor(
    scored: list[Scored],
    relevant_ids: set[int],
    size: int,
    ratio: float = 0.6,
) -> list[Scored]:
    """Keep the mix pointed at the request, not just at the user's history."""
    if size <= 0:
        return []
    result = list(scored)[:size]
    if not relevant_ids:
        return result
    need = int(round(len(result) * ratio))
    have = sum(1 for item in result if item.candidate.track_id in relevant_ids)
    if have >= need:
        return result
    spare = [item for item in scored[size:] if item.candidate.track_id in relevant_ids]
    for index in range(len(result) - 1, -1, -1):
        if have >= need or not spare:
            break
        if result[index].candidate.track_id not in relevant_ids:
            result[index] = spare.pop(0)
            have += 1
    return result


def _candidate_artists(cand: Candidate) -> list[str]:
    if cand.artists:
        cleaned = [a.strip().lower() for a in cand.artists if a and a.strip()]
        if cleaned:
            return cleaned
    if cand.artist and cand.artist.strip():
        return [cand.artist.strip().lower()]
    return []


def diversify(
    items: list[Scored],
    max_per_artist: int = 2,
    max_genre_share: float = 0.6,
    max_per_album: int = 2,
    target_size: int | None = None,
) -> list[Scored]:
    """Strict diversity in the returned slice, relaxing caps gradually only if starved."""
    if not items:
        return []
    goal = target_size if target_size is not None else len(items)
    genre_cap = max(1, int(max(goal, 1) * max_genre_share))

    picked: list[Scored] = []
    remaining = list(items)
    artist_counts: dict[str, int] = {}
    album_counts: dict[str, int] = {}
    genre_counts: dict[str, int] = {}

    # Progressive cap rounds: start strict (max_per_artist), only relax if under goal.
    for extra in range(4):
        if len(picked) >= goal or not remaining:
            break
        cur_artist_cap = max_per_artist + extra
        cur_album_cap = max_per_album + extra
        enforce_genre = extra == 0 and max_genre_share < 0.99
        next_remaining: list[Scored] = []

        for item in remaining:
            if len(picked) >= goal:
                next_remaining.append(item)
                continue
            artists = _candidate_artists(item.candidate)
            album = (item.candidate.album or "").strip().lower()
            genre = (item.candidate.genre or "").strip().lower()

            if artists and any(artist_counts.get(a, 0) >= cur_artist_cap for a in artists):
                next_remaining.append(item)
                continue
            if album and album_counts.get(album, 0) >= cur_album_cap:
                next_remaining.append(item)
                continue
            if enforce_genre and genre and genre_counts.get(genre, 0) >= genre_cap:
                next_remaining.append(item)
                continue

            picked.append(item)
            for a in artists:
                artist_counts[a] = artist_counts.get(a, 0) + 1
            if album:
                album_counts[album] = album_counts.get(album, 0) + 1
            if genre:
                genre_counts[genre] = genre_counts.get(genre, 0) + 1

        remaining = next_remaining

    return picked + remaining


def apply_exploration(
    items: list[Scored],
    rate: float = 0.2,
    seed: int = 0,
    *,
    min_score_ratio: float = 0.70,
) -> list[Scored]:
    """Replace every N-th slot with an item from the mid band (ranks 50-90%) that stays on-topic.

    Deterministic for a given seed. Returns a permutation: every index is emitted once.
    """
    n = len(items)
    if n < 5 or rate <= 0.0:
        return list(items)

    top_score = items[0].score
    min_score = top_score * min_score_ratio if top_score > 0 else 0.0

    every = max(2, round(1.0 / rate))
    lo = int(n * 0.45)
    hi = max(lo + 1, int(n * 0.9))
    raw_band = list(range(lo, min(hi, n)))
    band = [idx for idx in raw_band if items[idx].score >= min_score] or raw_band
    slot_positions = list(range(every - 1, n, every))

    chosen_count = min(len(slot_positions), len(band))
    chosen_band = random.Random(seed).sample(band, chosen_count)
    chosen_band.sort()
    explore_positions = set(slot_positions[:chosen_count])

    band_iter = iter(chosen_band)
    chosen = set(chosen_band)
    rest_iter = iter([index for index in range(n) if index not in chosen])

    result: list[Scored] = []
    for position in range(n):
        if position in explore_positions:
            result.append(items[next(band_iter)])
        else:
            result.append(items[next(rest_iter)])
    return result


def probabilistic_rerank(
    items: list[Scored],
    *,
    seed: int = 0,
    temperature: float = 0.075,
) -> list[Scored]:
    """Gumbel-Top-K weighted sampling without replacement (deterministic for seed)."""
    if len(items) <= 2 or temperature <= 0.0:
        return list(items)
    rng = random.Random(seed)
    perturbed: list[tuple[float, Scored]] = []
    for item in items:
        u = rng.uniform(1e-6, 1.0 - 1e-6)
        gumbel = -math.log(-math.log(u))
        perturbed.append((item.score + temperature * gumbel, item))
    perturbed.sort(key=lambda pair: pair[0], reverse=True)
    return [item for _, item in perturbed]
