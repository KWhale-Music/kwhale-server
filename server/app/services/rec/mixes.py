"""Mix assembly: query understanding -> multi-channel recall -> filter -> score -> diversify -> sequence."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from datetime import date
from typing import Any

import httpx

from ... import db
from ...config import settings
from ..genres import family_set
from ..normalize import dedup_artist_key
from . import candidates as candidates_service
from . import llm as llm_service
from . import profile as profile_service
from .features import cosine
from .llm import QueryPlan
from .scoring import (
    DEFAULT_PROFILE,
    Candidate,
    Scored,
    apply_exploration,
    candidate_families,
    diversify,
    enforce_relevance_floor,
    probabilistic_rerank,
    score_candidates,
)
from .sequencer import sequence_discovery, sequence_playlist

log = logging.getLogger("kwhale.rec.mixes")

CACHE_TTL = 900.0
_cache: dict[str, tuple[float, list[Scored]]] = {}
_counts: dict[str, int] = {}
_warming: set[int] = set()

_QEMB_CACHE_TTL = 3600.0
_QEMB_CACHE_MAX = 512
_qemb_cache: dict[str, tuple[float, list[float]]] = {}

KNOWN_KINDS = {"personal", "discovery", "genre", "artist", "radio", "mood"}

MOOD_TITLES = {
    "morning": "Утро",
    "focus": "Фокус",
    "evening_chill": "Вечер",
    "party": "Вечеринка",
    "night": "Ночь",
}


class UnknownMix(Exception):
    """Raised when a mix id is malformed or refers to a missing entity."""


def clear_cache() -> None:
    _cache.clear()
    _counts.clear()


def _count_key(user_id: int, mix_id: str) -> str:
    return f"{user_id}:{mix_id}"


async def _warm_counts(user_id: int, mix_ids: list[str]) -> None:
    try:
        for mix_id in mix_ids:
            if _count_key(user_id, mix_id) in _counts:
                continue
            try:
                await _assemble(user_id, mix_id)
            except Exception:  # noqa: BLE001 — one empty mix must not stop the rest
                continue
    finally:
        _warming.discard(user_id)


async def _load_profile_params(label: str = "default") -> dict:
    row = await db.fetch_one(
        "SELECT params FROM scoring_profiles WHERE label = :l",
        {"l": label},
    )
    merged = dict(DEFAULT_PROFILE)
    merged["filters"] = {}
    if row and row.get("params"):
        stored = dict(row["params"] or {})
        parts = {**DEFAULT_PROFILE["pref_parts"], **(stored.get("pref_parts") or {})}
        merged.update(stored)
        merged["pref_parts"] = parts
    return merged


def _seed(user_id: int, mix_id: str, prompt: str | None = None) -> int:
    norm_prompt = (prompt or "").strip().lower()
    raw = f"{user_id}:{date.today().isoformat()}:{mix_id}:{norm_prompt}".encode()
    return int(hashlib.sha256(raw).hexdigest()[:8], 16)


async def _artist_name(artist_id: int) -> str | None:
    row = await db.fetch_one("SELECT name FROM artists WHERE id = :id", {"id": artist_id})
    return (row or {}).get("name")


async def _embed_query(text: str) -> list[float] | None:
    """Embed a query string via Giga-Embeddings without importing heavy audio libs."""
    clean = (text or "").strip()
    if not clean or not settings.giga_embeddings_url:
        return None
    key = clean.lower()
    now = time.monotonic()
    cached = _qemb_cache.get(key)
    if cached and now - cached[0] < _QEMB_CACHE_TTL:
        return cached[1]
    try:
        async with httpx.AsyncClient(timeout=min(3.0, settings.giga_embeddings_timeout)) as client:
            resp = await client.post(
                settings.giga_embeddings_url,
                json={"input": [clean]},
            )
            resp.raise_for_status()
            vec = list(resp.json()["data"][0]["embedding"])
    except Exception:
        log.debug("query embedding unavailable for %r", clean[:60])
        return None
    if len(_qemb_cache) >= _QEMB_CACHE_MAX:
        oldest = min(_qemb_cache.items(), key=lambda kv: kv[1][0])[0]
        _qemb_cache.pop(oldest, None)
    _qemb_cache[key] = (now, vec)
    return vec


async def list_mixes(user_id: int) -> list[dict]:
    profile = await profile_service.load_profile(user_id)
    items: list[dict] = [
        {
            "id": "personal",
            "title": "Для тебя",
            "subtitle": "По твоим прослушиваниям",
            "kind": "personal",
            "track_count": 40,
        },
        {
            "id": "discovery",
            "title": "Открытия",
            "subtitle": "То, что ты давно не слушал",
            "kind": "discovery",
            "track_count": 40,
        },
    ]

    for preset, title in MOOD_TITLES.items():
        items.append(
            {
                "id": f"mood:{preset}",
                "title": title,
                "subtitle": "Плейлист настроения",
                "kind": "mood",
                "track_count": 30,
            }
        )

    # Only surface genres that actually have local playable tracks (>= 3)
    local_genre_rows = await db.fetch_all(
        """
        SELECT unnest(t.genres_norm) AS fam, COUNT(DISTINCT t.id) AS n
        FROM tracks t
        WHERE EXISTS (
            SELECT 1 FROM track_sources s
            WHERE s.track_id = t.id AND s.connector = 'local' AND s.available
        )
        GROUP BY 1
        HAVING COUNT(DISTINCT t.id) >= 3
        """
    )
    local_genres = {str(r["fam"]) for r in local_genre_rows}
    top_genres = [
        g for g, _ in sorted(profile["genres"].items(), key=lambda kv: kv[1], reverse=True)
        if not local_genres or g in local_genres
    ][:3]
    for genre in top_genres:
        items.append(
            {
                "id": f"genre:{genre}",
                "title": genre.title(),
                "subtitle": "Твой жанр",
                "kind": "genre",
                "track_count": 30,
            }
        )

    top_artists = sorted(profile["artists"].items(), key=lambda kv: kv[1], reverse=True)[:8]
    if top_artists:
        names = [name for name, _ in top_artists]
        rows = await db.fetch_all(
            """
            SELECT a.id, lower(a.name) AS lname, COUNT(DISTINCT t.id) AS local_n
            FROM artists a
            JOIN track_artists ta ON ta.artist_id = a.id
            JOIN tracks t ON t.id = ta.track_id
            WHERE lower(a.name) = ANY(:names)
              AND EXISTS (
                  SELECT 1 FROM track_sources s
                  WHERE s.track_id = t.id AND s.connector = 'local' AND s.available
              )
            GROUP BY a.id, lower(a.name)
            HAVING COUNT(DISTINCT t.id) >= 2
            """,
            {"names": names},
        )
        by_name = {str(r["lname"]): int(r["id"]) for r in rows}
        added_artists = 0
        for name, _ in top_artists:
            artist_id = by_name.get(name)
            if artist_id is None:
                continue
            items.append(
                {
                    "id": f"artist:{artist_id}",
                    "title": name.title(),
                    "subtitle": "Любимый артист",
                    "kind": "artist",
                    "track_count": 30,
                }
            )
            added_artists += 1
            if added_artists >= 3:
                break

    for item in items:
        item["track_count"] = _counts.get(
            _count_key(user_id, item["id"]), item["track_count"]
        )

    if user_id not in _warming:
        _warming.add(user_id)
        asyncio.create_task(_warm_counts(user_id, [i["id"] for i in items]))

    return items


def _candidate_families(cand: Candidate) -> set[str]:
    """Genre families incl. cross-genre tags (jazz-rap -> {rap, jazz})."""
    return candidate_families(cand)


def _style_match(cand: Candidate, plan: QueryPlan) -> bool:
    """Does the track satisfy the *style* part of the request?

    Priority matters: for «джаз» a `piano` texture on a rock track is not a
    style match, while a jazz-rap track with `jazz_samples` is.  The first
    non-empty signal defines the requested style.
    """
    if plan.include_genres:
        return bool(_candidate_families(cand) & set(plan.include_genres))
    if plan.target_subgenres:
        return bool(set(cand.subgenres or []) & set(plan.target_subgenres))
    if plan.target_textures:
        return bool(set(cand.textures or []) & set(plan.target_textures))
    if plan.target_moods:
        return bool(set(cand.moods or []) & set(plan.target_moods))
    if plan.target_scenes:
        return bool(set(cand.scenes or []) & set(plan.target_scenes))
    return True


def _style_share(cands: list[Candidate], plan: QueryPlan) -> int:
    return sum(1 for cand in cands if _style_match(cand, plan))


def _subgenre_families(tag: str) -> set[str]:
    return set(family_set([str(tag).replace("_", " ")]))


def _candidate_artist_keys(cand: Candidate) -> set[str]:
    names = list(cand.artists or [])
    if cand.artist:
        names.append(cand.artist)
    return {dedup_artist_key(n) for n in names if dedup_artist_key(n)}


def _apply_filters(cands: list[Candidate], filters: dict[str, Any]) -> list[Candidate]:
    """Filter candidates against legacy or QueryPlan-derived filter dicts.

    Strict semantics: numeric bounds reject candidates whose feature is `None`
    so unindexed tracks never slip past mood constraints.
    """
    if not filters:
        return cands

    out: list[Candidate] = []
    include = family_set(filters.get("genres_include", []))
    exclude = family_set(filters.get("genres_exclude", []))
    exc_subs = {str(s).strip().lower() for s in (filters.get("subgenres_exclude") or []) if s}
    exc_moods = {str(m).strip().lower() for m in (filters.get("moods_exclude") or []) if m}
    exc_textures = {str(t).strip().lower() for t in (filters.get("textures_exclude") or []) if t}
    exc_artists = {dedup_artist_key(str(a)) for a in (filters.get("exclude_artists") or []) if str(a).strip()}
    target_lang = filters.get("language")
    year_min = filters.get("year_min")
    year_max = filters.get("year_max")

    for cand in cands:
        families = _candidate_families(cand)
        if include and not (families & include):
            continue
        if exclude:
            hit = families & exclude
            if hit:
                # «Без рэпа» removes pure rap only.  A jazz-rap / pop-rap record
                # (rap + jazz/pop) survives, because the user keeps those.
                pure_rap_exemption = not (hit - {"rap"}) and bool(families - {"rap"})
                if not pure_rap_exemption:
                    continue
        if exc_subs:
            hit_subs = set(cand.subgenres or []) & exc_subs
            if hit_subs:
                # «Без рэпа» also emits rap subgenres (russian_rap, trap…).
                # Apply the same pure-rap exemption: a track with a non-rap
                # family survives, unless the *specific* excluded subgenre is
                # itself a non-pure-rap style (e.g. jazz_rap).
                exempt = "rap" in exclude and bool(families - {"rap"})
                hard_subs = (
                    {s for s in hit_subs if _subgenre_families(s) != {"rap"}}
                    if exempt
                    else hit_subs
                )
                if hard_subs:
                    continue
        if exc_moods and (set(cand.moods or []) & exc_moods):
            continue
        if exc_textures and (set(cand.textures or []) & exc_textures):
            continue
        if exc_artists:
            c_keys = _candidate_artist_keys(cand)
            if any(ea in ck or ck in ea for ea in exc_artists for ck in c_keys):
                continue
        if target_lang and cand.language is not None and cand.language != target_lang:
            continue
        if year_min is not None and cand.year is not None and cand.year < int(year_min):
            continue
        if year_max is not None and cand.year is not None and cand.year > int(year_max):
            continue

        if (value := filters.get("energy_min")) is not None:
            if cand.energy is None or cand.energy < float(value):
                continue
        if (value := filters.get("energy_max")) is not None:
            if cand.energy is None or cand.energy > float(value):
                continue
        if (value := filters.get("valence_min")) is not None:
            if cand.valence is None or cand.valence < float(value):
                continue
        if (value := filters.get("valence_max")) is not None:
            if cand.valence is None or cand.valence > float(value):
                continue
        if (value := filters.get("bpm_min")) is not None:
            if cand.bpm is None:
                continue
            # Octave-aware: 70 BPM half-time still matches >= 130 BPM trap if 2*bpm >= min
            eff_bpm = cand.bpm if cand.bpm >= float(value) else (cand.bpm * 2.0 if cand.bpm * 2.0 <= 210 else cand.bpm)
            if eff_bpm < float(value):
                continue
        if (value := filters.get("bpm_max")) is not None:
            if cand.bpm is None:
                continue
            eff_bpm = cand.bpm if cand.bpm <= float(value) else (cand.bpm * 0.5 if cand.bpm * 0.5 >= 55 else cand.bpm)
            if eff_bpm > float(value):
                continue
        if (value := filters.get("acousticness_min")) is not None:
            if cand.acousticness is None or cand.acousticness < float(value):
                continue
        if (value := filters.get("acousticness_max")) is not None:
            if cand.acousticness is None or cand.acousticness > float(value):
                continue
        if (value := filters.get("instrumentalness_min")) is not None:
            if cand.instrumentalness is None or cand.instrumentalness < float(value):
                continue
        out.append(cand)
    return out


def _hard_invariant_filters(plan: QueryPlan) -> dict[str, Any]:
    """Constraints that must NEVER be relaxed (explicit exclusions)."""
    out: dict[str, Any] = {}
    if plan.exclude_genres:
        out["genres_exclude"] = list(plan.exclude_genres)
    if plan.exclude_subgenres:
        out["subgenres_exclude"] = list(plan.exclude_subgenres)
    if plan.exclude_moods:
        out["moods_exclude"] = list(plan.exclude_moods)
    if plan.exclude_textures:
        out["textures_exclude"] = list(plan.exclude_textures)
    if plan.exclude_artists:
        out["exclude_artists"] = list(plan.exclude_artists)
    return out


def _strict_filters(plan: QueryPlan) -> dict[str, Any]:
    """Hard envelope for the first pass.

    `include_genres` is intentionally NOT here: the user prefers a fuller mix
    with ~1 in 10 adjacent tracks over a short "perfect" list.  Genre/style is a
    ranking boost (`query_relevance_score`), not a bouncer.
    """
    out = _hard_invariant_filters(plan)
    out.update(plan.hard_bounds)
    if plan.target_language:
        out["language"] = plan.target_language
    if plan.year_min is not None:
        out["year_min"] = plan.year_min
    if plan.year_max is not None:
        out["year_max"] = plan.year_max
    return out


def _widened_filters(plan: QueryPlan) -> dict[str, Any]:
    out = _hard_invariant_filters(plan)
    if plan.target_language:
        out["language"] = plan.target_language
    for k, v in plan.hard_bounds.items():
        if k.endswith("_min"):
            step = 18.0 if k.startswith("bpm") else 0.15
            out[k] = max(0.0, float(v) - step)
        elif k.endswith("_max"):
            step = 18.0 if k.startswith("bpm") else 0.15
            cap = 300.0 if k.startswith("bpm") else 1.0
            out[k] = min(cap, float(v) + step)
    return out


def _essential_filters(plan: QueryPlan) -> dict[str, Any]:
    """Fallback envelope: invariants + primary energy bounds, dropping sparse bounds."""
    out = _hard_invariant_filters(plan)
    if (e_min := plan.hard_bounds.get("energy_min")) is not None:
        out["energy_min"] = max(0.0, float(e_min) - 0.15)
    if (e_max := plan.hard_bounds.get("energy_max")) is not None:
        out["energy_max"] = min(1.0, float(e_max) + 0.15)
    return out


def _cap_recent(
    items: list[Scored],
    target_size: int,
    *,
    days: int = 3,
    share: float = 0.1,
) -> list[Scored]:
    """Keep at most ~`share` recently played tracks in the visible discovery head.

    Recent plays are heavily down-scored, but they stay eligible (the user
    prefers a full mix over a hard wall).  The cap prevents yesterday's
    favourites from filling the shelf.
    """
    head = list(items[:target_size])
    tail = list(items[target_size:])
    allowed = max(1, int(round(target_size * share)))
    recent = [x for x in head if candidates_service.within_days(x.candidate.last_played, days)]
    if len(recent) <= allowed:
        return items
    keep_ids = {id(x) for x in recent[:allowed]}
    kept = [
        x
        for x in head
        if not candidates_service.within_days(x.candidate.last_played, days) or id(x) in keep_ids
    ]
    overflow = [
        x
        for x in head
        if candidates_service.within_days(x.candidate.last_played, days) and id(x) not in keep_ids
    ]
    refill: list[Scored] = []
    rest: list[Scored] = []
    for item in tail:
        if len(kept) + len(refill) < target_size and not candidates_service.within_days(
            item.candidate.last_played, days
        ):
            refill.append(item)
        else:
            rest.append(item)
    return kept + refill + overflow + rest


def _attach_taste_vectors(cands: list[Candidate], profile: dict[str, Any], mood_preset: str | None = None) -> None:
    taste = profile.get("taste") or {}
    taste_vec = taste.get("taste")
    recent_vec = taste.get("recent") or taste_vec
    skip_vec = taste.get("skip")
    mood_vec = (taste.get("moods") or {}).get(mood_preset) if mood_preset else None

    for cand in cands:
        if cand.vec is None:
            continue
        base_sim = cosine(cand.vec, taste_vec)
        recent_sim = cosine(cand.vec, recent_vec)
        sims = [value for value in (base_sim, recent_sim) if value is not None]
        if sims:
            cand.sim_audio = sum(sims) / len(sims)
        cand.skip_sim = cosine(cand.vec, skip_vec) if skip_vec else None
        if mood_vec:
            cand.mood_pack_sim = cosine(cand.vec, mood_vec)


async def _assemble(
    user_id: int,
    mix_id: str,
    prompt: str | None = None,
    api_key: str | None = None,
    endpoint: str | None = None,
    model: str | None = None,
    *,
    plan_override: QueryPlan | None = None,
    target_size: int | None = None,
) -> list[Scored]:
    key = f"{user_id}:{mix_id}:{prompt or ''}:{target_size or ''}"
    if plan_override is None:
        cached = _cache.get(key)
        now = time.monotonic()
        if cached and now - cached[0] < CACHE_TTL:
            return cached[1]
    else:
        now = time.monotonic()

    kind, _, value = mix_id.partition(":")
    if kind not in KNOWN_KINDS:
        raise UnknownMix(mix_id)

    profile = await profile_service.load_profile(user_id)
    base = await _load_profile_params()
    seed = _seed(user_id, mix_id, prompt)

    plan: QueryPlan | None = plan_override
    effective_kind = kind
    mood_preset: str | None = None

    if prompt and plan is None:
        plan = await llm_service.plan_for(prompt, api_key=api_key, endpoint=endpoint, model=model)
        effective_kind = "prompt"
    elif kind == "mood":
        if value not in llm_service.PRESETS:
            raise UnknownMix(mix_id)
        mood_preset = value
        if plan is None:
            plan = llm_service.plan_for_mood(value)
    elif kind == "discovery":
        base.update({"pref": 0.45, "ctx": 0.10, "novelty": 0.45})

    if plan is not None and target_size is not None and target_size > 0:
        plan = plan.model_copy(update={"size": max(10, min(100, target_size))})

    query_vec: list[float] | None = None
    if plan is not None and plan.semantic_query:
        query_vec = await _embed_query(plan.semantic_query)

    if kind == "radio":
        try:
            seed_id = int(value)
        except ValueError as exc:
            raise UnknownMix(mix_id) from exc
        cands = await _assemble_similar_candidates(user_id, profile, seed_id)
        _attach_taste_vectors(cands, profile)
        radio_profile = {
            "pref": 0.85,
            "ctx": 0.05,
            "novelty": 0.10,
            "pref_parts": {
                "audio": 0.45,
                "text": 0.30,
                "cooc": 0.15,
                "genre": 0.05,
                "artist": 0.05,
            },
        }
        scored = score_candidates(cands, radio_profile, profile.get("context") or {})
        scored = diversify(scored, max_per_artist=2, max_genre_share=0.75, target_size=50)
        scored = sequence_playlist(scored[:50], arc="wave", seed=seed)
        _cache[key] = (now, scored)
        _counts[_count_key(user_id, mix_id)] = len(scored)
        return scored

    cands, _cooc = await candidates_service.collect_for_request(
        user_id,
        profile,
        kind=effective_kind,
        value=value,
        plan=plan,
        query_vec=query_vec,
        limit=600,
        seed=seed,
    )

    if kind == "genre" and not prompt:
        target_families = family_set([value])
        cands = [
            c for c in cands
            if (_candidate_families(c) & target_families)
            or (c.genre or "").lower() == value.lower()
        ]
    elif kind == "artist" and not prompt:
        try:
            artist_id = int(value)
        except ValueError as exc:
            raise UnknownMix(mix_id) from exc
        name = await _artist_name(artist_id)
        if not name:
            raise UnknownMix(mix_id)
        target_key = dedup_artist_key(name)
        cands = [c for c in cands if target_key in _candidate_artist_keys(c)]

    _attach_taste_vectors(cands, profile, mood_preset=mood_preset)

    if plan is not None:
        # Tiered safe filtering — NEVER drops hard exclusions!
        # Genre/style is a ranking preference, not a hard filter: a full list
        # with ~1 in 10 adjacent tracks beats a short "perfect" list.
        need_style = max(1, int(round(plan.size * 0.8)))
        strict = _strict_filters(plan)
        filtered = _apply_filters(cands, strict)
        if len(filtered) < plan.size or _style_share(filtered, plan) < need_style:
            widened = _apply_filters(cands, _widened_filters(plan))
            if len(widened) > len(filtered):
                filtered = widened
        if len(filtered) < plan.size or _style_share(filtered, plan) < need_style:
            essential = _apply_filters(cands, _essential_filters(plan))
            if len(essential) > len(filtered):
                filtered = essential
        if len(filtered) < plan.size or _style_share(filtered, plan) < need_style:
            hard_only = _apply_filters(cands, _hard_invariant_filters(plan))
            if len(hard_only) > len(filtered):
                filtered = hard_only

        scored = score_candidates(
            filtered,
            base,
            profile.get("context") or {},
            mode="prompt",
            plan=plan,
        )
        relevant = {
            item.candidate.track_id
            for item in scored
            if _style_match(item.candidate, plan)
        }
        if relevant:
            scored = enforce_relevance_floor(
                scored, relevant, size=min(len(scored), plan.size), ratio=0.8
            )
        cap_genre = 1.0 if len(plan.include_genres) == 1 else float(base.get("max_genre_share", 0.6))
        scored = diversify(
            scored,
            max_per_artist=int(base.get("max_per_artist", 2)),
            max_genre_share=cap_genre,
            target_size=plan.size,
        )
        # Apply exploration only within high-relevance corridor of the selected mix
        head = scored[: max(plan.size, min(len(scored), plan.size + 10))]
        head = apply_exploration(head, rate=plan.exploration, seed=seed, min_score_ratio=0.75)
        scored = sequence_playlist(head[: plan.size], arc=plan.arc, seed=seed) + scored[len(head) :]
    else:
        score_mode = "discovery" if kind == "discovery" else "personal"
        history_days = float(profile.get("history_days") or 60.0)
        scored = score_candidates(
            cands,
            base,
            profile.get("context") or {},
            mode=score_mode,
            history_days=history_days,
        )
        target_sz = 40
        cap_genre = 1.0 if kind in ("genre", "artist") else float(base.get("max_genre_share", 0.6))
        artist_cap = 20 if kind == "artist" else int(base.get("max_per_artist", 2))
        if kind == "discovery":
            top_pool = probabilistic_rerank(scored[: min(len(scored), 90)], seed=seed, temperature=0.075)
            # With large favourite libraries the "liked but never played" stream
            # outranks every played track. Reserve the anchor share structurally:
            # the familiar-entry points must be present in the shelf, not just
            # hoped for in the score order.
            anchor_reasons = ("familiar_anchor", "rediscover")
            anchors = [x for x in scored if x.reason_code in anchor_reasons]
            n_anchor = min(len(anchors), max(1, round(target_sz * 0.25))) if anchors else 0
            chosen_anchors = anchors[:n_anchor]
            chosen_ids = {id(x) for x in chosen_anchors}
            pool = chosen_anchors + [
                x for x in (top_pool + scored[min(len(scored), 90) :]) if id(x) not in chosen_ids
            ]
            scored = diversify(
                pool,
                max_per_artist=artist_cap,
                max_genre_share=cap_genre,
                target_size=target_sz,
            )
            head = sequence_discovery(scored[:target_sz], seed=seed, anchor_ratio=0.25)
            scored = _cap_recent(head + scored[target_sz:], target_sz, days=3, share=0.1)
        else:
            scored = diversify(
                scored,
                max_per_artist=artist_cap,
                max_genre_share=cap_genre,
                target_size=target_sz,
            )
            scored = apply_exploration(
                scored,
                rate=float(base.get("exploration", 0.2)),
                seed=seed,
                min_score_ratio=0.55,
            )
            if kind != "artist":
                head = sequence_playlist(scored[:target_sz], arc="wave", seed=seed)
                scored = head + scored[target_sz:]

    _cache[key] = (now, scored)
    if not prompt and plan_override is None:
        _counts[_count_key(user_id, mix_id)] = len(scored)
    return scored


async def _assemble_similar_candidates(
    user_id: int, profile: dict[str, Any], track_id: int
) -> list[Candidate]:
    audio_ids = await candidates_service.audio_neighbours(user_id, track_id, limit=120)
    text_ids = await candidates_service.text_neighbours(user_id, track_id, limit=120)
    cooc_ids = await candidates_service.neighbours(track_id, limit=100, user_id=user_id)
    fallback_ids = (
        await candidates_service.same_artist_or_genre(user_id, track_id, limit=100)
        if len(audio_ids) + len(text_ids) < 20
        else []
    )

    seen: set[int] = set()
    unique: list[int] = []
    for tid in audio_ids + text_ids + cooc_ids + fallback_ids:
        if tid != track_id and tid not in seen:
            seen.add(tid)
            unique.append(tid)

    cands = await candidates_service.hydrate(user_id, profile, unique[:250])
    seed_list = await candidates_service.hydrate(user_id, profile, [track_id])
    seed_cand = seed_list[0] if seed_list else None

    audio_rank = {tid: 1.0 - (idx / max(1, len(audio_ids))) for idx, tid in enumerate(audio_ids)}
    text_rank = {tid: 1.0 - (idx / max(1, len(text_ids))) for idx, tid in enumerate(text_ids)}
    cooc_set = set(cooc_ids)

    seed_tags = set(
        (seed_cand.subgenres or []) + (seed_cand.moods or []) + (seed_cand.textures or [])
    ) if seed_cand else set()

    for cand in cands:
        if seed_cand and seed_cand.vec and cand.vec:
            cand.sim_audio = cosine(seed_cand.vec, cand.vec)
        elif cand.track_id in audio_rank:
            cand.sim_audio = 2.0 * audio_rank[cand.track_id] - 1.0

        if cand.track_id in text_rank:
            cand.sim_text = 2.0 * text_rank[cand.track_id] - 1.0
        if cand.track_id in cooc_set:
            cand.cooc = 0.85
        if seed_tags:
            c_tags = set((cand.subgenres or []) + (cand.moods or []) + (cand.textures or []))
            overlap = len(seed_tags & c_tags)
            if overlap:
                cand.ref_sim = min(1.0, overlap / max(1.0, len(seed_tags)))
    return cands


async def build(
    user_id: int,
    mix_id: str,
    limit: int,
    offset: int = 0,
    prompt: str | None = None,
) -> list[Scored]:
    scored = await _assemble(user_id, mix_id, prompt)
    if not scored:
        return []
    return scored[offset : offset + limit]


async def prompt_mix(
    user_id: int,
    prompt: str,
    size: int,
    api_key: str | None = None,
    endpoint: str | None = None,
    model: str | None = None,
) -> list[Scored]:
    scored = await _assemble(
        user_id, "personal", prompt, api_key, endpoint, model, target_size=size
    )
    if not scored:
        return []
    return scored[:size]


async def similar(user_id: int, track_id: int, limit: int = 40) -> list[Scored]:
    profile = await profile_service.load_profile(user_id)
    cands = await _assemble_similar_candidates(user_id, profile, track_id)
    sim_profile = {
        "pref": 0.88,
        "ctx": 0.05,
        "novelty": 0.07,
        "pref_parts": {
            "audio": 0.45,
            "text": 0.30,
            "cooc": 0.15,
            "genre": 0.05,
            "artist": 0.05,
        },
    }
    scored = score_candidates(cands, sim_profile, profile.get("context") or {})
    scored = diversify(scored, max_per_artist=2, max_genre_share=0.8, target_size=limit)
    return sequence_playlist(scored[:limit], arc="steady", seed=track_id)


def reason_text(item: Scored) -> str:
    code = item.reason_code
    cand = item.candidate
    if code == "tag_match" and item.reason_detail:
        return f"По атмосфере и стилю: {item.reason_detail}"
    if code == "query_match":
        if cand.vibe_summary:
            short = cand.vibe_summary.split("|")[0].strip()
            return f"По запросу · {short[:65]}"
        return "Точное попадание в запрос"
    if code == "vibe_match":
        return "Подходит по темпу и настроению"
    if code == "liked_artist" and cand.artist:
        return f"Потому что ты слушаешь {cand.artist}"
    if code == "genre_match" and cand.genre:
        return f"Жанр: {cand.genre}"
    if code == "familiar_anchor":
        return f"Знакомый ориентир · {cand.artist}" if cand.artist else "Знакомый ориентир"
    if code == "bridge":
        detail = (item.reason_detail or cand.artist or "").strip()
        return f"Мост к новому · {detail}" if detail else "Похоже по звучанию"
    if code == "liked_unheard":
        return "В избранном, но ещё не слушал"
    if code == "recent_play":
        return "Ты недавно это слушал"
    if code == "rediscover":
        return "Давно не слушал" if cand.plays > 0 else "Из твоих плейлистов"
    if code == "similar_sound":
        return "Похоже по звучанию"
    if code == "similar_lyrics":
        return "Похоже по смыслу и настроению"
    if code == "explore":
        return "В тему из каталога"
    return "Новое для тебя"
