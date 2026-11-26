"""Tests for the query planner and legacy LLM patch layer (pure, no DB)."""

from __future__ import annotations

from app.services.rec.llm import (
    MOOD_PLANS,
    PRESETS,
    merge_profile,
    patch_from_prompt,
    plan_from_prompt_fallback,
    validate_patch,
    validate_plan,
)


def test_fallback_jazz_uses_style_tags_not_only_genre():
    plan = plan_from_prompt_fallback("спокойный джаз на вечер, без рэпа")
    assert plan.include_genres == ["jazz"]
    assert plan.exclude_genres == ["rap"]
    assert "jazz_rap" in plan.target_subgenres
    assert "jazz_samples" in plan.target_textures


def test_validate_clamps_and_normalises():
    patch = validate_patch({"preset": "party", "weights": {"pref": 5, "ctx": -1, "novelty": 0.5}})
    assert abs(sum(patch["weights"].values()) - 1.0) < 1e-9
    assert all(0.0 <= value <= 1.0 for value in patch["weights"].values())


def test_unknown_preset_falls_back():
    assert validate_patch({"preset": "nonsense"})["preset"] == "default"


def test_exploration_and_size_are_clamped():
    patch = validate_patch({"exploration": 9, "size": 500})
    assert patch["exploration"] == 0.3
    assert patch["size"] == 100


def test_keyword_fallback_picks_preset():
    assert patch_from_prompt("спокойное на вечер")["preset"] == "evening_chill"
    assert patch_from_prompt("энергичная тренировка")["filters"]["energy_min"] == 0.6


def test_merge_profile_applies_weights():
    base = dict(PRESETS["default"])
    patch = validate_patch({"preset": "focus"})
    merged = merge_profile(base, patch)
    assert merged["pref"] == patch["weights"]["pref"]
    assert merged["novelty"] == patch["weights"]["novelty"]


def test_plan_fallback_compound_styles_and_exclusions():
    hard_rap = plan_from_prompt_fallback("жёсткий русский рэп")
    assert hard_rap.include_genres == ["rap"]
    assert hard_rap.target_language == "ru"
    assert "aggressive" in hard_rap.target_moods
    assert hard_rap.hard_bounds.get("energy_min") == 0.55

    ac_rap = plan_from_prompt_fallback("акустико-гитарный рэп")
    assert "acoustic_rap" in ac_rap.target_subgenres
    assert "acoustic_guitar" in ac_rap.target_textures
    assert "rap" in ac_rap.include_genres

    jazz_rap = plan_from_prompt_fallback("рэп с джазовыми сэмплами")
    assert "jazz_rap" in jazz_rap.target_subgenres
    assert "jazz_samples" in jazz_rap.target_textures

    jazz_no_rap = plan_from_prompt_fallback("спокойный джаз на вечер, без рэпа")
    assert "jazz" in jazz_no_rap.include_genres
    assert "rap" in jazz_no_rap.exclude_genres
    assert jazz_no_rap.preset == "evening_chill"
    assert jazz_no_rap.hard_bounds.get("energy_max") is not None

    acoustic_only = plan_from_prompt_fallback("спокойная музыка под акустическую гитару")
    assert "acoustic_guitar" in acoustic_only.target_textures
    assert "rap" in acoustic_only.exclude_genres

    drive = plan_from_prompt_fallback("ехать по ночной трассе под дождём, на закате")
    assert {"night_drive", "rainy", "sunset"} <= set(drive.target_scenes)

    instr = plan_from_prompt_fallback("грустная музыка без слов 2019")
    assert instr.target_language == "instrumental"
    assert "sad" in instr.target_moods
    assert instr.year_min == 2017 and instr.year_max == 2021

    ref_exc = plan_from_prompt_fallback("треки как oxxxymiron без instasamka")
    assert "oxxxymiron" in ref_exc.reference_artists
    assert "instasamka" in ref_exc.exclude_artists


def test_mood_plans_are_valid_and_distinct():
    assert set(MOOD_PLANS) == {"morning", "focus", "evening_chill", "party", "night"}
    plans = {name: validate_plan(cfg) for name, cfg in MOOD_PLANS.items()}
    assert plans["party"].audio_target["energy"] > plans["focus"].audio_target["energy"]
    assert plans["morning"].audio_target["valence"] > plans["night"].audio_target["valence"]
    assert plans["focus"].hard_bounds.get("energy_max", 1.0) <= 0.65
