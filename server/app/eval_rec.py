"""Offline quality & benchmark evaluation for the recommendation layer.

    python -m app.eval_rec                     # global event metrics
    python -m app.eval_rec --user 1            # + holdout for one user
    python -m app.eval_rec --user 1 --prompt "жёсткий русский рэп"
    python -m app.eval_rec --user 1 --benchmark # full audit benchmark matrix
"""

from __future__ import annotations

import argparse
import asyncio
import json
from datetime import UTC, datetime
from typing import Any

from . import db, httpcore


def hit_rate(holdout_ids: list[int], ranked_ids: list[int], k: int) -> float:
    """Share of holdout tracks that appear in the top-K of the ranking."""
    if not holdout_ids:
        return 0.0
    top = set(ranked_ids[: max(0, k)])
    return sum(1 for track_id in holdout_ids if track_id in top) / len(holdout_ids)


def mean_percentile(holdout_ids: list[int], ranked_ids: list[int]) -> float | None:
    """Average position of the holdout tracks (0 = top, 1 = bottom)."""
    if not holdout_ids or not ranked_ids:
        return None
    positions = {track_id: index for index, track_id in enumerate(ranked_ids)}
    percentiles = [
        positions[track_id] / max(1, len(ranked_ids) - 1)
        for track_id in holdout_ids
        if track_id in positions
    ]
    if not percentiles:
        return None
    return sum(percentiles) / len(percentiles)


def jaccard(a: list[int], b: list[int]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 0.0
    return len(sa & sb) / max(1, len(sa | sb))


async def event_metrics(days: int = 30) -> dict[str, Any]:
    rows = await db.fetch_all(
        """
        SELECT COALESCE(NULLIF(payload->>'surface', ''), 'unknown') AS surface,
               COUNT(*) FILTER (WHERE type = 'complete') AS completes,
               COUNT(*) FILTER (WHERE type = 'skip')     AS skips,
               COUNT(*) FILTER (WHERE type = 'play')     AS plays,
               COUNT(*)                                  AS total
        FROM events
        WHERE received_at > NOW() - (:d * INTERVAL '1 day')
        GROUP BY 1
        ORDER BY total DESC
        """,
        {"d": int(days)},
    )
    out: dict[str, Any] = {}
    for row in rows:
        starts = int(row["plays"] or 0) + int(row["completes"] or 0) + int(row["skips"] or 0)
        out[row["surface"]] = {
            "events": int(row["total"]),
            "plays": int(row["plays"] or 0),
            "completes": int(row["completes"] or 0),
            "skips": int(row["skips"] or 0),
            "skip_rate": round(int(row["skips"] or 0) / starts, 3) if starts else 0.0,
            "complete_rate": round(int(row["completes"] or 0) / starts, 3) if starts else 0.0,
        }
    return out


async def prompt_relevance(user_id: int, prompt: str, size: int = 20) -> dict[str, Any]:
    from .services.genres import family_set
    from .services.rec import llm as llm_service
    from .services.rec import mixes as mixes_service

    plan = await llm_service.plan_for(prompt)
    inc_families = family_set(plan.include_genres)
    exc_families = family_set(plan.exclude_genres)
    scored = await mixes_service._assemble(user_id, "personal", prompt, target_size=size)
    items = scored[:size]

    relevant = 0
    excluded_leaks = 0
    with_features = 0
    energies: list[float] = []
    valences: list[float] = []
    bpms: list[float] = []
    artists: set[str] = set()
    wanted_tags = set(plan.target_subgenres + plan.target_textures + plan.target_moods + plan.target_scenes)
    tag_hits = 0

    for item in items:
        c = item.candidate
        fams = mixes_service._candidate_families(c)
        if not inc_families or (fams & inc_families):
            relevant += 1
        if exc_families and (fams & exc_families):
            excluded_leaks += 1
        if c.has_features:
            with_features += 1
        if c.energy is not None:
            energies.append(c.energy)
        if c.valence is not None:
            valences.append(c.valence)
        if c.bpm is not None:
            bpms.append(c.bpm)
        if c.artist:
            artists.add(c.artist.lower())
        have_tags = set((c.subgenres or []) + (c.textures or []) + (c.moods or []) + (c.scenes or []))
        if wanted_tags and (have_tags & wanted_tags):
            tag_hits += 1

    n = len(items)
    return {
        "prompt": prompt,
        "source": plan.source,
        "include_genres": sorted(inc_families),
        "exclude_genres": sorted(exc_families),
        "size": n,
        "relevant": relevant,
        "relevant_share": round(relevant / n, 3) if n else 0.0,
        "excluded_leaks": excluded_leaks,
        "feature_share": round(with_features / n, 3) if n else 0.0,
        "tag_match_share": round(tag_hits / n, 3) if n and wanted_tags else None,
        "unique_artists": len(artists),
        "avg_energy": round(sum(energies) / len(energies), 3) if energies else None,
        "avg_valence": round(sum(valences) / len(valences), 3) if valences else None,
        "avg_bpm": round(sum(bpms) / len(bpms), 1) if bpms else None,
        "track_ids": [item.candidate.track_id for item in items],
    }


async def benchmark_suite(user_id: int) -> dict[str, Any]:
    """Run the 10 audit prompts + discovery + mood packs check."""
    from .services.rec import mixes as mixes_service

    prompts = [
        "жёсткий русский рэп",
        "агрессивный русский рэп",
        "тёмный фонк",
        "спокойный джаз на вечер, без рэпа",
        "весёлая поп для вечеринки",
        "нежная акустика под гитару",
        "русский рэп 2019",
        "музыка под которую приятно ехать в дождь в закат по трассе с девушкой",
        "акустико-гитарный рэп",
        "рэп с джазовыми сэмплами",
    ]
    prompt_results: list[dict[str, Any]] = []
    by_prompt: dict[str, list[int]] = {}
    for p in prompts:
        res = await prompt_relevance(user_id, p, size=20)
        by_prompt[p] = res.pop("track_ids")
        prompt_results.append(res)

    jacc = jaccard(
        by_prompt["жёсткий русский рэп"],
        by_prompt["агрессивный русский рэп"],
    )

    disc = await mixes_service._assemble(user_id, "discovery")
    now = datetime.now(UTC)
    days_list: list[float] = []
    for item in disc[:20]:
        if item.candidate.last_played is not None:
            lp = item.candidate.last_played
            lp_tz = lp if lp.tzinfo else lp.replace(tzinfo=UTC)
            days_list.append((now - lp_tz).total_seconds() / 86400.0)
    days_list.sort()
    median_days = round(days_list[len(days_list) // 2], 1) if days_list else None

    moods_summary: dict[str, Any] = {}
    for preset in ("morning", "focus", "evening_chill", "party", "night"):
        m_items = (await mixes_service._assemble(user_id, f"mood:{preset}"))[:20]
        e_vals = [x.candidate.energy for x in m_items if x.candidate.energy is not None]
        v_vals = [x.candidate.valence for x in m_items if x.candidate.valence is not None]
        moods_summary[preset] = {
            "count": len(m_items),
            "avg_energy": round(sum(e_vals) / len(e_vals), 3) if e_vals else None,
            "avg_valence": round(sum(v_vals) / len(v_vals), 3) if v_vals else None,
        }

    return {
        "prompts": prompt_results,
        "jaccard_hard_vs_aggressive": round(jacc, 3),
        "discovery": {
            "count": len(disc),
            "played_in_top20": len(days_list),
            "median_days_since_play": median_days,
        },
        "moods": moods_summary,
    }


async def holdout(user_id: int, k: int = 20) -> dict[str, Any]:
    from .services.rec import candidates as candidates_service
    from .services.rec import profile as profile_service
    from .services.rec.features import cosine
    from .services.rec.scoring import DEFAULT_PROFILE, score_candidates

    rows = await db.fetch_all(
        """
        SELECT track_id FROM mv_user_track_stats
        WHERE user_id = :u AND completes > 0 AND last_played IS NOT NULL
        ORDER BY last_played DESC
        LIMIT :k
        """,
        {"u": user_id, "k": k},
    )
    holdout_ids = [int(r["track_id"]) for r in rows]
    if not holdout_ids:
        return {"user": user_id, "holdout": 0, "note": "нет истории дослушиваний"}

    profile = await profile_service.load_profile(user_id)
    pool_ids = set(holdout_ids)
    collected = await candidates_service.collect(user_id, profile, limit=500)
    pool_ids.update(candidate.track_id for candidate in collected)
    cands = await candidates_service.hydrate(user_id, profile, sorted(pool_ids))
    for cand in cands:
        if cand.vec is not None and profile.get("taste"):
            cand.sim_audio = cosine(cand.vec, profile["taste"].get("taste"))
    scored = score_candidates(cands, DEFAULT_PROFILE, profile.get("context") or {})
    ranked = [item.candidate.track_id for item in scored]
    return {
        "user": user_id,
        "holdout": len(holdout_ids),
        "pool": len(ranked),
        "hit_at_20": round(hit_rate(holdout_ids, ranked, 20), 3),
        "mean_percentile": (
            round(value, 3) if (value := mean_percentile(holdout_ids, ranked)) is not None else None
        ),
        "note": "материализованные статы не отыгрываются — метрика с утечкой, только sanity check",
    }


async def _cli() -> None:
    parser = argparse.ArgumentParser(description="KWhale recommendation quality")
    parser.add_argument("--user", type=int, default=None)
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--days", type=int, default=30)
    args = parser.parse_args()

    await db.apply_schema()
    await httpcore.load_proxy_settings()
    result: dict[str, Any] = {"events": await event_metrics(days=args.days)}
    try:
        if args.user is not None:
            result["holdout"] = await holdout(args.user)
            if args.prompt:
                pr = await prompt_relevance(args.user, args.prompt)
                pr.pop("track_ids", None)
                result["prompt_relevance"] = pr
            if args.benchmark:
                result["benchmark"] = await benchmark_suite(args.user)
    finally:
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
        await httpcore.aclose_all()
        await db.close_engine()


if __name__ == "__main__":
    asyncio.run(_cli())
