"""Unit tests for the pure scoring core (no database required)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from app.services.rec.llm import plan_from_prompt_fallback
from app.services.rec.scoring import (
    DEFAULT_PROFILE,
    Candidate,
    Scored,
    apply_exploration,
    diversify,
    novelty_score,
    pref_score,
    pref_terms,
    probabilistic_rerank,
    score_candidates,
)
from app.services.rec.sequencer import sequence_discovery


def test_pref_renormalises_missing_terms():
    candidate = Candidate(track_id=1, artist_affinity=1.0)
    value, reason = pref_score(candidate, {"artist": 0.35, "genre": 0.20, "cooc": 0.25})
    assert value == 1.0
    assert reason == "liked_artist"


def test_explicit_zero_lowers_the_score():
    candidate = Candidate(track_id=1, artist_affinity=1.0, genre_affinity=0.0, cooc=0.0)
    value, _ = pref_score(candidate, {"artist": 0.35, "genre": 0.20, "cooc": 0.25})
    assert abs(value - 0.4375) < 1e-9


def test_pref_without_signals_is_neutral():
    value, reason = pref_score(Candidate(track_id=1), DEFAULT_PROFILE["pref_parts"])
    assert value == 0.0
    assert reason == "new_for_you"


def test_novelty_never_played_is_max():
    assert novelty_score(Candidate(track_id=1)) == 1.0


def test_novelty_personal_cooldown_recovers_while_discovery_boosts_old():
    now = datetime(2026, 11, 6, tzinfo=UTC)
    recent = Candidate(track_id=1, plays=1, last_played=now - timedelta(hours=6))
    mid = Candidate(track_id=2, plays=1, last_played=now - timedelta(days=14))
    old = Candidate(track_id=3, plays=1, last_played=now - timedelta(days=90))
    older = Candidate(track_id=4, plays=1, last_played=now - timedelta(days=180))

    assert novelty_score(recent, now, mode="personal") < 0.2
    assert 0.65 <= novelty_score(mid, now, mode="personal") <= 0.80
    assert novelty_score(old, now, mode="personal") >= 0.90

    # Discovery suppresses ultra-fresh plays (<3d), ramps smoothly through cutoff,
    # adapts for newer accounts (14d), and peaks for forgotten (>30d) tracks.
    assert novelty_score(recent, now, mode="discovery") < 0.05
    mid_mature = novelty_score(mid, now, mode="discovery", history_days=90.0)
    mid_new_user = novelty_score(mid, now, mode="discovery", history_days=14.0)
    assert 0.25 < mid_mature < mid_new_user
    assert 0.62 <= novelty_score(old, now, mode="discovery") < novelty_score(older, now, mode="discovery")


def test_novelty_repeat_penalty_lowers_but_does_not_zero():
    now = datetime(2026, 11, 6, tzinfo=UTC)
    candidate = Candidate(track_id=1, plays=5, last_played=now - timedelta(days=1))
    value = novelty_score(candidate, now)
    assert 0.0 < value < 0.8


def test_engagement_does_not_saturate_on_two_plays():
    light = Candidate(track_id=1, plays=1, completes=1)
    devoted = Candidate(track_id=2, plays=5, completes=5, repeats=2, liked=True)
    heavy_skip = Candidate(track_id=3, plays=4, completes=0, skips=4)
    assert pref_terms(light)["engagement"] < 0.7
    assert pref_terms(devoted)["engagement"] > pref_terms(light)["engagement"]
    assert pref_terms(heavy_skip)["engagement"] < 0.2


def test_score_is_bounded_and_sorted():
    now = datetime(2026, 11, 6, tzinfo=UTC)
    candidates = [
        Candidate(track_id=1, artist_affinity=1.0, plays=0),
        Candidate(track_id=2, artist_affinity=0.1, plays=8, last_played=now - timedelta(days=2)),
        Candidate(track_id=3, artist_affinity=0.5, plays=2, last_played=now - timedelta(days=40)),
    ]
    scored = score_candidates(candidates, DEFAULT_PROFILE, {}, now)
    assert len(scored) == 3
    assert all(0.0 <= item.score <= 1.0 for item in scored)
    assert scored == sorted(scored, key=lambda item: item.score, reverse=True)
    assert scored[0].candidate.track_id == 1


def test_diversify_caps_artist():
    cands = [Candidate(track_id=i, artist="Aphex Twin", artist_affinity=1.0) for i in range(6)]
    cands.extend(Candidate(track_id=10 + i, artist=f"Other {i}", artist_affinity=0.7) for i in range(4))
    items = score_candidates(cands, DEFAULT_PROFILE)
    picked = diversify(items, max_per_artist=2, max_genre_share=1.0, target_size=6)
    assert len(picked) == 10
    top_6_aphex = [item for item in picked[:6] if item.candidate.artist == "Aphex Twin"]
    assert len(top_6_aphex) == 2


def test_query_plan_prefers_matching_subgenre_and_texture():
    plan = plan_from_prompt_fallback("акустико-гитарный рэп")
    acoustic_rap = Candidate(
        track_id=1,
        genres=["rap"],
        subgenres=["acoustic_rap"],
        textures=["acoustic_guitar", "acoustic"],
        energy=0.45,
        acousticness=0.82,
        has_features=True,
    )
    hard_trap = Candidate(
        track_id=2,
        genres=["rap"],
        subgenres=["trap", "rage"],
        textures=["808_bass", "distorted"],
        energy=0.88,
        acousticness=0.05,
        has_features=True,
    )
    scored = score_candidates([hard_trap, acoustic_rap], DEFAULT_PROFILE, mode="prompt", plan=plan)
    assert scored[0].candidate.track_id == 1


def test_exploration_is_deterministic_and_returns_permutation():
    items = score_candidates(
        [Candidate(track_id=i, artist=f"a{i}", artist_affinity=1.0 - i / 100) for i in range(20)],
        DEFAULT_PROFILE,
    )
    first = [item.candidate.track_id for item in apply_exploration(items, rate=0.2, seed=42)]
    second = [item.candidate.track_id for item in apply_exploration(items, rate=0.2, seed=42)]
    assert first == second
    assert sorted(first) == list(range(20))
    explore_positions = (4, 9, 14, 19)
    band = set(range(10, 18))
    assert any(first[position] in band for position in explore_positions)


def test_exploration_noop_for_small_lists():
    items = score_candidates([Candidate(track_id=i) for i in range(3)], DEFAULT_PROFILE)
    assert [item.candidate.track_id for item in apply_exploration(items)] == [0, 1, 2]


def test_probabilistic_rerank_and_discovery_anchor_bridges():
    items = score_candidates(
        [
            Candidate(track_id=i, artist=f"A{i}", artist_affinity=0.8 - i * 0.01, plays=0)
            for i in range(15)
        ],
        DEFAULT_PROFILE,
        mode="discovery",
    )
    a1 = [x.candidate.track_id for x in probabilistic_rerank(items, seed=101, temperature=0.08)]
    a2 = [x.candidate.track_id for x in probabilistic_rerank(items, seed=101, temperature=0.08)]
    b1 = [x.candidate.track_id for x in probabilistic_rerank(items, seed=202, temperature=0.08)]
    assert a1 == a2
    assert sorted(a1) == sorted(b1)
    assert a1 != b1

    mixed = [
        Scored(candidate=Candidate(track_id=100, artist="Anchor1", liked=True, plays=2, completes=2, vec=[1.0, 0.0]), score=0.72, reason_code="familiar_anchor"),
        Scored(candidate=Candidate(track_id=101, artist="Anchor2", liked=True, plays=1, completes=1, vec=[0.0, 1.0]), score=0.70, reason_code="familiar_anchor"),
        Scored(candidate=Candidate(track_id=200, artist="NewA", plays=0, vec=[0.95, 0.05]), score=0.80, reason_code="new_for_you"),
        Scored(candidate=Candidate(track_id=201, artist="NewB", plays=0, vec=[0.90, 0.10]), score=0.79, reason_code="new_for_you"),
        Scored(candidate=Candidate(track_id=202, artist="NewC", plays=0, vec=[0.85, 0.15]), score=0.78, reason_code="new_for_you"),
        Scored(candidate=Candidate(track_id=203, artist="NewD", plays=0, vec=[0.10, 0.90]), score=0.77, reason_code="new_for_you"),
        Scored(candidate=Candidate(track_id=204, artist="NewE", plays=0, vec=[0.05, 0.95]), score=0.76, reason_code="new_for_you"),
        Scored(candidate=Candidate(track_id=205, artist="NewF", plays=0, vec=[0.20, 0.80]), score=0.75, reason_code="new_for_you"),
    ]
    seq = sequence_discovery(mixed, seed=7, anchor_ratio=0.25, bridge_len=3)
    assert len(seq) == len(mixed)
    assert seq[0].candidate.track_id == 100
    # After Anchor1 ([1, 0]), closest acoustic discoveries come first before Anchor2
    assert {seq[1].candidate.track_id, seq[2].candidate.track_id, seq[3].candidate.track_id} == {200, 201, 202}
    assert seq[4].candidate.track_id == 101


def test_liked_unheard_is_not_a_discovery_anchor():
    items = [
        Scored(
            candidate=Candidate(track_id=1, artist="LikedNeverPlayed", liked=True, plays=0, vec=[1.0, 0.0]),
            score=0.95,
            reason_code="liked_unheard",
        ),
        Scored(
            candidate=Candidate(track_id=2, artist="RealAnchor", liked=True, plays=3, completes=2, vec=[1.0, 0.0]),
            score=0.70,
            reason_code="familiar_anchor",
        ),
        Scored(candidate=Candidate(track_id=3, artist="New1", plays=0, vec=[0.9, 0.1]), score=0.80, reason_code="new_for_you"),
        Scored(candidate=Candidate(track_id=4, artist="New2", plays=0, vec=[0.8, 0.2]), score=0.79, reason_code="new_for_you"),
        Scored(candidate=Candidate(track_id=5, artist="New3", plays=0, vec=[0.7, 0.3]), score=0.78, reason_code="new_for_you"),
    ]
    seq = sequence_discovery(items, seed=1, anchor_ratio=0.25, bridge_len=3)
    assert seq[0].candidate.track_id == 2  # only the played track may lead


def test_recent_play_never_leads_a_discovery_shelf():
    items = [
        Scored(candidate=Candidate(track_id=1, artist="Recent", plays=2, completes=1), score=0.99, reason_code="recent_play"),
        Scored(candidate=Candidate(track_id=2, artist="Anchor", plays=4, completes=3), score=0.60, reason_code="familiar_anchor"),
        Scored(candidate=Candidate(track_id=3, artist="New1", plays=0), score=0.70, reason_code="new_for_you"),
        Scored(candidate=Candidate(track_id=4, artist="New2", plays=0), score=0.69, reason_code="new_for_you"),
        Scored(candidate=Candidate(track_id=5, artist="New3", plays=0), score=0.68, reason_code="new_for_you"),
    ]
    seq = sequence_discovery(items, seed=1, anchor_ratio=0.25, bridge_len=3)
    assert seq[0].candidate.track_id != 1
    assert seq[-1].candidate.track_id == 1
