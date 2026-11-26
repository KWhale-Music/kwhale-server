"""Recommendation targeting: genre families, feature vectors, relevance floor.

The feedback («жёсткий русский рэп» returned Алена Швец) traced to three
gaps: providers' raw genre strings never compared equal, the ranker saw only
energy/valence of ~20 stored features, and the prompt filter was dropped
entirely when it starved the pool.
"""

from __future__ import annotations

import math

import pytest

from app.services.genres import FAMILY_KEYS, family_set, genre_families, normalize_genres
from app.services.rec.features import cosine, feature_vector
from app.services.rec.llm import patch_from_prompt, validate_patch
from app.services.rec.mixes import _apply_filters
from app.services.rec.scoring import (
    Candidate,
    Scored,
    candidate_families,
    enforce_relevance_floor,
    pref_terms,
    score_candidates,
)

# ── genre families ───────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Hip-Hop & Rap", ["rap"]),
        ("rusrap", ["rap"]),
        ("хип-хоп", ["rap"]),
        ("русский рэп", ["rap"]),
        ("trap", ["rap"]),
        ("phonk", ["rap"]),
        ("R&B/Soul", ["rnb"]),
        ("шансон", ["folk"]),
        ("Drum and Bass", ["electronic"]),
        ("lo-fi beats", ["instrumental"]),
        ("", []),
        (None, []),
    ],
)
def test_genre_families(raw, expected):
    assert genre_families(raw) == expected


def test_normalize_genres_unions_families():
    assert normalize_genres(["Hip-Hop & Rap", "pop"]) == ["rap", "pop"]
    assert normalize_genres(["synthpop"]) == ["pop"]


def test_family_set_maps_prompt_phrases():
    assert family_set(["русский рэп"]) == {"rap"}
    assert family_set(["жёсткий рэп"]) == {"rap"}
    assert family_set([]) == set()


def test_family_keys_are_canonical_vocabulary():
    assert "rap" in FAMILY_KEYS and "rnb" in FAMILY_KEYS


# ── audio-feature vectors ────────────────────────────────────────────────────


def _feature_row(**overrides):
    row = {
        "bpm": 140.0,
        "beat_confidence": 0.9,
        "loudness_lufs": -8.0,
        "dynamic_complexity": 0.4,
        "danceability": 0.7,
        "onset_rate": 3.0,
        "spectral_centroid": 2500.0,
        "spectral_rolloff": 7000.0,
        "spectral_flux": 500.0,
        "mfcc_mean": [10.0, -5.0, 3.0, 0.0, 8.0],
        "energy": 0.8,
        "valence": 0.3,
        "voice_instrumental": 0.9,
    }
    row.update(overrides)
    return row


def test_feature_vector_is_mean_centered_and_aligned():
    vec = feature_vector(_feature_row())
    assert vec is not None
    assert len(vec) == 22
    assert all(-1.0 <= value <= 1.0 for value in vec)
    assert math.isclose(sum(v * v for v in vec), 1.0, rel_tol=1e-5)


def test_feature_vector_requires_enough_signal():
    assert feature_vector({}) is None
    assert feature_vector(None) is None
    sparse = {"bpm": 120.0, "energy": 0.5}
    assert feature_vector(sparse) is None


def test_cosine_similarity_basics():
    a = feature_vector(_feature_row())
    b = feature_vector(_feature_row())
    c = feature_vector(_feature_row(energy=0.05, valence=0.95, bpm=60.0))
    assert cosine(a, b) == pytest.approx(1.0)
    assert cosine(a, c) is not None and cosine(a, c) < 0.9
    assert cosine(None, a) is None
    assert cosine(a, [0.0] * 22) is None


# ── prompt patch mapping ─────────────────────────────────────────────────────


def test_prompt_patch_maps_families_for_hard_russian_rap():
    patch = patch_from_prompt("жёсткий русский рэп")
    assert patch["filters"].get("genres_include") == ["rap"]


def test_prompt_patch_honours_bez_exclusion():
    patch = patch_from_prompt("спокойный джаз на вечер, без рэпа")
    assert patch["filters"].get("genres_include") == ["jazz"]
    assert patch["filters"].get("genres_exclude") == ["rap"]


def test_llm_patch_normalizes_raw_genre_strings():
    patch = validate_patch(
        {"preset": "party", "filters": {"genres_include": ["Hip-Hop & Rap", "trap"]}}
    )
    assert patch["filters"]["genres_include"] == ["rap"]


# ── filters match families ───────────────────────────────────────────────────


def _candidate(track_id: int, genres: list[str], energy: float | None = 0.8) -> Candidate:
    return Candidate(track_id=track_id, genres=genres, energy=energy)


def test_filters_match_canonical_families():
    cands = [
        _candidate(1, ["rap"]),
        _candidate(2, ["pop"]),
        _candidate(3, ["rnb"]),
    ]
    assert [c.track_id for c in _apply_filters(cands, {"genres_include": ["rap"]})] == [1]
    assert {c.track_id for c in _apply_filters(cands, {"genres_exclude": ["pop"]})} == {1, 3}


def test_candidate_families_infers_jazz_from_samples():
    cand = Candidate(track_id=1, genres=["rap"], textures=["jazz_samples"])
    assert candidate_families(cand) == {"rap", "jazz"}


def test_no_rap_removes_pure_rap_but_keeps_jazz_rap_and_pop_rap():
    cands = [
        Candidate(track_id=1, genres=["rap"], genre="Rap"),
        Candidate(track_id=2, genres=["rap"], genre="Rap", textures=["jazz_samples"]),
        Candidate(track_id=3, genres=["pop"], genre="Pop", subgenres=["jazz_rap"]),
        Candidate(track_id=4, genres=["rap"], genre="Rap", subgenres=["russian_rap", "trap"]),
    ]
    out = _apply_filters(cands, {"genres_exclude": ["rap"]})
    assert {c.track_id for c in out} == {2, 3}


def test_no_rap_subgenres_keep_jazz_rap_but_drop_pure_rap():
    cands = [
        Candidate(track_id=1, genres=["rap"], subgenres=["russian_rap", "trap"]),
        Candidate(track_id=2, genres=["rap"], subgenres=["russian_rap"], textures=["jazz_samples"]),
        Candidate(track_id=3, genres=["rap"], subgenres=["jazz_rap"]),
    ]
    out = _apply_filters(
        cands,
        {"genres_exclude": ["rap"], "subgenres_exclude": ["russian_rap", "trap", "jazz_rap"]},
    )
    assert {c.track_id for c in out} == {2}


def test_filters_keep_energy_envelope_and_reject_none():
    cands = [
        _candidate(1, ["rap"], energy=0.9),
        _candidate(2, ["rap"], energy=0.2),
        _candidate(3, ["rap"], energy=None),
    ]
    out = _apply_filters(cands, {"genres_include": ["rap"], "energy_min": 0.6})
    assert [c.track_id for c in out] == [1]


def test_filters_enforce_bpm_and_exclude_artists():
    cands = [
        Candidate(track_id=1, artist="Oxxxymiron", artists=["Oxxxymiron"], bpm=140.0),
        Candidate(track_id=2, artist="Artist B", artists=["Artist B", "LSP"], bpm=70.0),
        Candidate(track_id=3, artist="Slow C", artists=["Slow C"], bpm=55.0),
        Candidate(track_id=4, artist="Clean D", artists=["Clean D"], bpm=136.0),
    ]
    out = _apply_filters(cands, {"bpm_min": 130.0, "exclude_artists": ["Oxxxymiron", "LSP"]})
    assert [c.track_id for c in out] == [4]


def test_genre_phonk_and_word_boundaries():
    assert genre_families("фонк") == ["rap"]
    assert genre_families("ruspop") == ["pop"]
    assert genre_families("dub") == ["reggae"]
    assert genre_families("dubstep") == ["electronic"]
    assert "instrumental" not in genre_families("podcast show")


# ── implicit feedback and relevance floor ───────────────────────────────────


def test_engagement_term_uses_completes_and_skips():
    engaged = Candidate(track_id=1, plays=4, completes=2, skips=0)
    dropped = Candidate(track_id=2, plays=4, completes=0, skips=4)
    assert pref_terms(engaged).get("engagement", 0) > pref_terms(dropped).get("engagement", 0)


def test_repeatedly_skipped_track_is_suppressed():
    kept = Candidate(track_id=1, plays=3, completes=1)
    dropped = Candidate(track_id=2, plays=5, completes=0, skips=3)
    scored = score_candidates([kept, dropped], {"pref": 1.0, "ctx": 0.0, "novelty": 0.0})
    assert scored[0].candidate.track_id == 1
    assert scored[0].score > scored[1].score


def test_relevance_floor_keeps_mix_on_target():
    scored = [
        Scored(candidate=Candidate(track_id=index), score=1.0 - index / 10, reason_code="x")
        for index in range(1, 11)
    ]
    relevant = {8, 9, 10}  # only the bottom three match the request
    out = enforce_relevance_floor(scored, relevant, size=10, ratio=0.3)
    kept = {item.candidate.track_id for item in out}
    assert len([track for track in kept if track in relevant]) == 3
