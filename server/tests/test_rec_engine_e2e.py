"""End-to-end integration tests for the rebuilt recommendation engine (Postgres + pgvector)."""

from __future__ import annotations

import json
import math
import uuid
from datetime import UTC, datetime, timedelta

import pytest

from app import db, eval_rec
from app.services.genres import normalize_genres
from app.services.rec import audio_embed, llm, mixes, stats, tagger
from app.services.rec.features import feature_vector, load_corpus_stats
from app.services.rec.queue import MODEL_VERSIONS
from app.services.rec.scoring import DEFAULT_PROFILE, Candidate, Scored
from app.services.rec.sequencer import sequence_playlist


def _basis_vec(weights: dict[int, float], dim: int = 1024) -> list[float]:
    vec = [0.0] * dim
    for idx, val in weights.items():
        vec[idx % dim] = float(val)
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def _vec_literal(vec: list[float]) -> str:
    return "[" + ",".join(f"{v:.6f}" for v in vec) + "]"


async def _seed_rec_catalog() -> tuple[int, dict[str, int]]:
    await db.apply_schema()
    await db.execute(
        "TRUNCATE users, tracks, artists, playlists, events, user_onboarding, rec_feature_stats, user_mood_profiles CASCADE"
    )
    mixes.clear_cache()

    urows = await db.execute_returning(
        "INSERT INTO users (username, password_hash) VALUES ('rec_tester', 'x') RETURNING id"
    )
    uid = int(urows[0]["id"])

    # Seed realistic tracks across styles, textures, moods, scenes, local vs remote, broken vs valid
    catalog = [
        {
            "key": "hard_rap_oxxy",
            "title": "Город под подошвой",
            "artists": ["Oxxxymiron"],
            "album": "Горгород",
            "year": 2015,
            "genre": "rusrap",
            "local": True,
            "energy": 0.88,
            "valence": 0.25,
            "bpm": 138.0,
            "danceability": 0.72,
            "acousticness": 0.08,
            "instrumentalness": 0.02,
            "subgenres": ["russian_rap", "boom_bap"],
            "moods": ["aggressive", "intense", "dark"],
            "textures": ["heavy_bass", "live_drums", "raspy_vocals"],
            "scenes": ["workout", "night_drive"],
            "language": "ru",
            "basis": {0: 1.0, 5: 0.4},
            "summary": "жёсткий техничный русский рэп | aggressive dark russian rap",
        },
        {
            "key": "hard_rap_obladaet",
            "title": "PLAYERS CLUB",
            "artists": ["OBLADAET"],
            "album": "PLAYERS CLUB",
            "year": 2021,
            "genre": "Hip-Hop & Rap",
            "local": True,
            "energy": 0.90,
            "valence": 0.30,
            "bpm": 144.0,
            "danceability": 0.78,
            "acousticness": 0.04,
            "instrumentalness": 0.01,
            "subgenres": ["russian_rap", "drill", "trap"],
            "moods": ["aggressive", "intense", "dark", "party"],
            "textures": ["808_bass", "heavy_bass", "distorted"],
            "scenes": ["workout", "party", "club"],
            "language": "ru",
            "basis": {0: 0.95, 6: 0.5},
            "summary": "жёсткий русский дрилл и трэп | aggressive drill trap heavy 808",
        },
        {
            "key": "hard_rap_pharaoh",
            "title": "Дико, например",
            "artists": ["PHARAOH"],
            "album": "Pink Phloyd",
            "year": 2017,
            "genre": "rusrap",
            "local": True,
            "energy": 0.84,
            "valence": 0.22,
            "bpm": 134.0,
            "danceability": 0.70,
            "acousticness": 0.05,
            "instrumentalness": 0.02,
            "subgenres": ["russian_rap", "trap", "cloud_rap"],
            "moods": ["dark", "aggressive", "intense", "nocturnal"],
            "textures": ["808_bass", "distorted", "synth"],
            "scenes": ["night_drive", "party"],
            "language": "ru",
            "basis": {0: 0.9, 7: 0.5},
            "summary": "мрачный жёсткий трэп | dark aggressive russian trap",
        },
        {
            "key": "acoustic_rap",
            "title": "Выдыхай",
            "artists": ["Noize MC"],
            "album": "Последний альбом",
            "year": 2010,
            "genre": "rusrap",
            "local": True,
            "energy": 0.44,
            "valence": 0.42,
            "bpm": 88.0,
            "danceability": 0.48,
            "acousticness": 0.85,
            "instrumentalness": 0.05,
            "subgenres": ["acoustic_rap", "russian_rap"],
            "moods": ["melancholic", "warm", "nostalgic"],
            "textures": ["acoustic_guitar", "acoustic", "clean_vocals"],
            "scenes": ["evening", "relax"],
            "language": "ru",
            "basis": {1: 1.0, 2: 0.4},
            "summary": "лирический акустико-гитарный рэп | acoustic guitar russian rap",
        },
        {
            "key": "jazz_rap",
            "title": "Интро с саксофоном",
            "artists": ["Krec"],
            "album": "Нет волшебства",
            "year": 2004,
            "genre": "Hip-Hop & Rap",
            "local": True,
            "energy": 0.48,
            "valence": 0.46,
            "bpm": 90.0,
            "danceability": 0.58,
            "acousticness": 0.58,
            "instrumentalness": 0.12,
            "subgenres": ["jazz_rap", "boom_bap", "chillhop"],
            "moods": ["warm", "chill", "nostalgic"],
            "textures": ["jazz_samples", "saxophone", "piano", "vinyl"],
            "scenes": ["evening", "rainy", "walk"],
            "language": "ru",
            "basis": {3: 0.9, 1: 0.5},
            "summary": "рэп с джазовыми сэмплами и саксофоном | jazz rap boom bap jazz samples",
        },
        {
            "key": "sad_pop_shvets",
            "title": "Мальчик из Питера",
            "artists": ["Алёна Швец"],
            "album": "Королева отстоя",
            "year": 2020,
            "genre": "indie",
            "local": True,
            "energy": 0.42,
            "valence": 0.32,
            "bpm": 102.0,
            "danceability": 0.46,
            "acousticness": 0.72,
            "instrumentalness": 0.02,
            "subgenres": ["indie_pop", "bedroom_pop"],
            "moods": ["melancholic", "sad", "tender"],
            "textures": ["acoustic_guitar", "clean_vocals"],
            "scenes": ["rainy", "evening"],
            "language": "ru",
            "basis": {2: 0.85, 8: 0.4},
            "summary": "меланхоличный инди-поп под гитару | melancholic bedroom indie pop",
        },
        {
            "key": "calm_jazz",
            "title": "Blue in Green",
            "artists": ["Miles Davis"],
            "album": "Kind of Blue",
            "year": 1959,
            "genre": "Jazz",
            "local": True,
            "energy": 0.24,
            "valence": 0.40,
            "bpm": 74.0,
            "danceability": 0.35,
            "acousticness": 0.92,
            "instrumentalness": 0.94,
            "subgenres": ["jazz", "nu_jazz"],
            "moods": ["calm", "chill", "warm", "nocturnal"],
            "textures": ["saxophone", "brass", "piano", "acoustic"],
            "scenes": ["evening", "late_night", "focus", "rainy"],
            "language": "instrumental",
            "basis": {4: 1.0, 10: 0.4},
            "summary": "спокойный ночной инструментальный джаз | calm nocturnal instrumental jazz",
        },
        {
            "key": "calm_guitar",
            "title": "River Acoustic",
            "artists": ["Nick Drake"],
            "album": "Pink Moon",
            "year": 1972,
            "genre": "folk",
            "local": True,
            "energy": 0.26,
            "valence": 0.44,
            "bpm": 82.0,
            "danceability": 0.32,
            "acousticness": 0.95,
            "instrumentalness": 0.45,
            "subgenres": ["acoustic_folk", "indie_folk"],
            "moods": ["calm", "tender", "warm", "melancholic"],
            "textures": ["acoustic_guitar", "acoustic", "clean_vocals"],
            "scenes": ["evening", "relax", "rainy", "focus"],
            "language": "en",
            "basis": {2: 1.0, 9: 0.4},
            "summary": "спокойная музыка под акустическую гитару | calm acoustic guitar folk",
        },
        {
            "key": "drive_sunset_synth",
            "title": "Nightcall",
            "artists": ["Kavinsky"],
            "album": "OutRun",
            "year": 2010,
            "genre": "Electronic",
            "local": True,
            "energy": 0.56,
            "valence": 0.36,
            "bpm": 91.0,
            "danceability": 0.62,
            "acousticness": 0.08,
            "instrumentalness": 0.25,
            "subgenres": ["synthwave", "retrowave"],
            "moods": ["atmospheric", "nocturnal", "dreamy", "warm"],
            "textures": ["synth", "pad", "heavy_bass"],
            "scenes": ["night_drive", "road_trip", "sunset", "rainy"],
            "language": "en",
            "basis": {11: 1.0, 12: 0.5},
            "summary": "ночная поездка по трассе на закате под дождём | night drive sunset rainy synthwave",
        },
        {
            "key": "morning_indie",
            "title": "Walking on a Dream",
            "artists": ["Empire of the Sun"],
            "album": "Walking on a Dream",
            "year": 2008,
            "genre": "Indie Pop",
            "local": True,
            "energy": 0.72,
            "valence": 0.84,
            "bpm": 122.0,
            "danceability": 0.74,
            "acousticness": 0.15,
            "instrumentalness": 0.05,
            "subgenres": ["synthpop", "indie_pop"],
            "moods": ["uplifting", "bright", "happy", "energetic"],
            "textures": ["synth", "clean_vocals"],
            "scenes": ["morning", "walk", "road_trip"],
            "language": "en",
            "basis": {13: 1.0},
            "summary": "светлый бодрый утренний синти-поп | bright uplifting morning synthpop",
        },
        {
            "key": "party_edm",
            "title": "Titanium Club",
            "artists": ["David Guetta", "Sia"],
            "album": "Nothing but the Beat",
            "year": 2011,
            "genre": "Dance & Electronic",
            "local": True,
            "energy": 0.92,
            "valence": 0.72,
            "bpm": 126.0,
            "danceability": 0.86,
            "acousticness": 0.04,
            "instrumentalness": 0.10,
            "subgenres": ["house", "dance_pop"],
            "moods": ["party", "energetic", "euphoric"],
            "textures": ["synth", "heavy_bass"],
            "scenes": ["party", "club", "workout"],
            "language": "en",
            "basis": {14: 1.0},
            "summary": "танцевальный клубный хаус для вечеринки | party club energetic house",
        },
        {
            "key": "remote_only",
            "title": "Remote Rap Song",
            "artists": ["Remote MC"],
            "album": "Cloud",
            "year": 2022,
            "genre": "rusrap",
            "local": False,
            "energy": 0.90,
            "valence": 0.30,
            "bpm": 140.0,
            "danceability": 0.75,
            "acousticness": 0.05,
            "instrumentalness": 0.01,
            "subgenres": ["russian_rap"],
            "moods": ["aggressive"],
            "textures": ["808_bass"],
            "scenes": ["workout"],
            "language": "ru",
            "basis": {0: 1.0},
            "summary": "жёсткий русский рэп только в облаке",
        },
        {
            "key": "broken_local",
            "title": "Corrupted Local Track",
            "artists": ["Broken Artist"],
            "album": "Broken",
            "year": 2021,
            "genre": "rusrap",
            "local": True,
            "broken": True,
            "energy": 0.88,
            "valence": 0.30,
            "bpm": 140.0,
            "danceability": 0.75,
            "acousticness": 0.05,
            "instrumentalness": 0.01,
            "subgenres": ["russian_rap"],
            "moods": ["aggressive"],
            "textures": ["808_bass"],
            "scenes": ["workout"],
            "language": "ru",
            "basis": {0: 1.0},
            "summary": "битый локальный файл",
        },
        {
            "key": "disliked_rap",
            "title": "Blocked Track",
            "artists": ["Blocked MC"],
            "album": "Blocked",
            "year": 2021,
            "genre": "rusrap",
            "local": True,
            "disliked": True,
            "energy": 0.90,
            "valence": 0.30,
            "bpm": 140.0,
            "danceability": 0.75,
            "acousticness": 0.05,
            "instrumentalness": 0.01,
            "subgenres": ["russian_rap"],
            "moods": ["aggressive"],
            "textures": ["808_bass"],
            "scenes": ["workout"],
            "language": "ru",
            "basis": {0: 1.0},
            "summary": "дизлайкнутый трек",
        },
        {
            "key": "old_forgotten_jazz",
            "title": "Autumn Leaves Old",
            "artists": ["Bill Evans"],
            "album": "Portrait in Jazz",
            "year": 1960,
            "genre": "Jazz",
            "local": True,
            "energy": 0.28,
            "valence": 0.45,
            "bpm": 78.0,
            "danceability": 0.38,
            "acousticness": 0.90,
            "instrumentalness": 0.95,
            "subgenres": ["jazz"],
            "moods": ["calm", "warm", "melancholic"],
            "textures": ["piano", "acoustic"],
            "scenes": ["focus", "evening"],
            "language": "instrumental",
            "basis": {4: 0.95, 10: 0.3},
            "summary": "старый спокойный джаз трио | calm piano jazz",
        },
        {
            "key": "hard_phonk",
            "title": "Shadow Phonk",
            "artists": ["Kordhell"],
            "album": "Phonk Tape",
            "year": 2022,
            "genre": "фонк",
            "local": True,
            "energy": 0.87,
            "valence": 0.22,
            "bpm": 132.0,
            "danceability": 0.80,
            "acousticness": 0.03,
            "instrumentalness": 0.70,
            "subgenres": ["phonk", "dark_phonk"],
            "moods": ["dark", "aggressive", "intense", "nocturnal"],
            "textures": ["808_bass", "heavy_bass", "distorted", "lo_fi"],
            "scenes": ["night_drive", "workout"],
            "language": "instrumental",
            "basis": {0: 0.6, 15: 0.8},
            "summary": "тёмный агрессивный фонк для ночной езды | dark phonk heavy bass",
        },
    ]

    ids: dict[str, int] = {}
    corpus_stats = await load_corpus_stats()
    for item in catalog:
        fam = normalize_genres([item["genre"]])
        rows = await db.execute_returning(
            """
            INSERT INTO tracks (title, artists, album, duration_sec, year, genre, genres_norm, last_seen_at)
            VALUES (:t, :a, :al, 180, :y, :g, :gn, NOW())
            RETURNING id
            """,
            {
                "t": item["title"],
                "a": item["artists"],
                "al": item["album"],
                "y": item["year"],
                "g": item["genre"],
                "gn": fam,
            },
        )
        tid = int(rows[0]["id"])
        ids[item["key"]] = tid

        for aname in item["artists"]:
            arow = await db.fetch_one("SELECT id FROM artists WHERE name = :n", {"n": aname})
            if not arow:
                arow = (
                    await db.execute_returning(
                        "INSERT INTO artists (name) VALUES (:n) RETURNING id", {"n": aname}
                    )
                )[0]
            await db.execute(
                "INSERT INTO track_artists (track_id, artist_id) VALUES (:t, :a) ON CONFLICT DO NOTHING",
                {"t": tid, "a": int(arow["id"])},
            )

        if item["local"]:
            srows = await db.execute_returning(
                "INSERT INTO track_sources (track_id, connector, external_id, file_path, available) "
                "VALUES (:t, 'local', :e, :p, TRUE) RETURNING id",
                {"t": tid, "e": f"local_{tid}", "p": f"/music/{tid}.flac"},
            )
            if item.get("broken"):
                await db.execute(
                    "INSERT INTO track_quality (track_id, local_source_id, file_path, error) "
                    "VALUES (:t, :s, :p, 'ffprobe invalid data')",
                    {"t": tid, "s": int(srows[0]["id"]), "p": f"/music/{tid}.flac"},
                )
        else:
            await db.execute(
                "INSERT INTO track_sources (track_id, connector, external_id, available) "
                "VALUES (:t, 'yandex', :e, TRUE)",
                {"t": tid, "e": f"ym_{tid}"},
            )

        if item.get("disliked"):
            await db.execute(
                "INSERT INTO dislikes (user_id, track_id) VALUES (:u, :t)",
                {"u": uid, "t": tid},
            )

        extra = {
            "acoustic_proxy": item["acousticness"],
            "bass_ratio": 0.65 if item["energy"] > 0.7 else 0.25,
            "mid_ratio": 0.45,
            "spectral_flatness": 0.12 if item["energy"] > 0.7 else 0.04,
            "spectral_contrast": 0.62,
            "zero_crossing_rate": 0.09 if item["energy"] > 0.7 else 0.03,
            "dissonance": 0.48 if "aggressive" in item["moods"] else 0.28,
            "pitch_salience": 0.62,
        }
        mfcc = [40.0, 20.0 if item["energy"] > 0.7 else 55.0, -5.0, 12.0, -6.0]
        feat_row = {
            "loudness_lufs": -6.0 if item["energy"] > 0.7 else -14.0,
            "dynamic_complexity": 3.0 if item["energy"] > 0.7 else 6.0,
            "bpm": item["bpm"],
            "beat_confidence": 0.85 if item["energy"] > 0.6 else 0.45,
            "danceability": item["danceability"],
            "onset_rate": 5.0 if item["energy"] > 0.7 else 2.2,
            "spectral_centroid": 2800.0 if item["energy"] > 0.7 else 1400.0,
            "spectral_rolloff": 6000.0 if item["energy"] > 0.7 else 2800.0,
            "spectral_flux": 4.0 if item["energy"] > 0.7 else 1.1,
            "energy": item["energy"],
            "valence": item["valence"],
            "mfcc_mean": mfcc,
            "dsp_extra": extra,
        }
        await db.execute(
            """
            INSERT INTO track_features (
                track_id, bpm, beat_confidence, loudness_lufs, dynamic_complexity,
                danceability, onset_rate, spectral_centroid, spectral_rolloff, spectral_flux,
                mfcc_mean, energy, valence, voice_instrumental, dsp_extra,
                subgenres, moods, textures, scenes, language, vibe_summary,
                acousticness, instrumentalness, tags_source, tags_version
            )
            VALUES (
                :tid, :bpm, :bc, :lufs, :dyn,
                :dance, :onset, :cent, :roll, :flux,
                :mfcc, :en, :val, :vi, CAST(:extra AS jsonb),
                :sub, :mood, :tex, :scn, :lang, :sum,
                :ac, :inst, 'llm', :tver
            )
            """,
            {
                "tid": tid,
                "bpm": feat_row["bpm"],
                "bc": feat_row["beat_confidence"],
                "lufs": feat_row["loudness_lufs"],
                "dyn": feat_row["dynamic_complexity"],
                "dance": feat_row["danceability"],
                "onset": feat_row["onset_rate"],
                "cent": feat_row["spectral_centroid"],
                "roll": feat_row["spectral_rolloff"],
                "flux": feat_row["spectral_flux"],
                "mfcc": mfcc,
                "en": item["energy"],
                "val": item["valence"],
                "vi": 1.0 - item["instrumentalness"],
                "extra": json.dumps(extra),
                "sub": item["subgenres"],
                "mood": item["moods"],
                "tex": item["textures"],
                "scn": item["scenes"],
                "lang": item["language"],
                "sum": item["summary"],
                "ac": item["acousticness"],
                "inst": item["instrumentalness"],
                "tver": MODEL_VERSIONS["tags"],
            },
        )
        avec = feature_vector(feat_row, corpus_stats)
        assert avec is not None
        padded = audio_embed.pad_to_column(avec)
        await db.execute(
            "INSERT INTO track_emb_audio (track_id, model_version, vec) VALUES (:t, :mv, CAST(:v AS vector))",
            {"t": tid, "mv": MODEL_VERSIONS["audio"], "v": _vec_literal(padded)},
        )
        tvec = _basis_vec(item["basis"])
        await db.execute(
            "INSERT INTO track_emb_text (track_id, model_version, vec) VALUES (:t, :mv, CAST(:v AS vector))",
            {"t": tid, "mv": MODEL_VERSIONS["text"], "v": _vec_literal(tvec)},
        )

    # User likes Oxxxymiron (unplayed) + recently played sad_pop_shvets + forgot old_forgotten_jazz 90d ago
    await db.execute(
        "INSERT INTO favorites (user_id, track_id) VALUES (:u, :t)",
        {"u": uid, "t": ids["hard_rap_oxxy"]},
    )
    now = datetime.now(UTC)
    for tid, ev_type, ts, payload in (
        (ids["sad_pop_shvets"], "play", now - timedelta(hours=3), {"surface": "mix", "context": {"mixId": "evening_chill"}}),
        (ids["sad_pop_shvets"], "complete", now - timedelta(hours=3), {"surface": "mix", "context": {"mixId": "evening_chill"}, "listenedMs": 175000, "trackDurationMs": 180000}),
        (ids["old_forgotten_jazz"], "play", now - timedelta(days=95), {"surface": "home"}),
        (ids["old_forgotten_jazz"], "complete", now - timedelta(days=95), {"surface": "home", "listenedMs": 178000, "trackDurationMs": 180000}),
        (ids["party_edm"], "play", now - timedelta(days=2), {"surface": "mix", "context": {"mixId": "mood:party"}}),
        (ids["party_edm"], "complete", now - timedelta(days=2), {"surface": "mix", "context": {"mixId": "mood:party"}, "listenedMs": 175000, "trackDurationMs": 180000}),
    ):
        await db.execute(
            "INSERT INTO events (id, user_id, track_id, type, payload, occurred_at) "
            "VALUES (:id, :u, :t, :ty, CAST(:p AS jsonb), :ts)",
            {"id": str(uuid.uuid4()), "u": uid, "t": tid, "ty": ev_type, "p": json.dumps(payload), "ts": ts},
        )
    await db.execute(
        "INSERT INTO user_onboarding (user_id, genres, artists) VALUES (:u, ARRAY['jazz', 'rap'], ARRAY['Nick Drake'])",
        {"u": uid},
    )
    await stats.refresh_stats()
    return uid, ids


@pytest.fixture
async def rec_catalog():
    try:
        yield await _seed_rec_catalog()
    finally:
        try:
            await db.execute(
                "TRUNCATE users, tracks, artists, playlists, events, user_onboarding, rec_feature_stats, user_mood_profiles CASCADE"
            )
        except Exception:
            pass
        await db.close_engine()


@pytest.mark.asyncio
async def test_prompt_mixes_and_exclusions_e2e(rec_catalog, monkeypatch):
    uid, ids = rec_catalog

    async def _no_llm(*_a, **_kw):
        return None

    monkeypatch.setattr(llm, "plan_from_llm", _no_llm)
    llm._plan_cache.clear()
    mixes._qemb_cache.clear()

    async def fake_embed(text: str) -> list[float] | None:
        low = text.lower()
        if "джаз" in low and ("рэп" in low or "jazz_rap" in low):
            return _basis_vec({3: 1.0, 1: 0.4})
        if "акуст" in low and ("рэп" in low or "acoustic_rap" in low):
            return _basis_vec({1: 1.0, 2: 0.3})
        if "рэп" in low or "russian_rap" in low:
            return _basis_vec({0: 1.0, 5: 0.4, 6: 0.4})
        if "джаз" in low or "jazz" in low:
            return _basis_vec({4: 1.0, 10: 0.4})
        if "гитар" in low or "acoustic" in low:
            return _basis_vec({2: 1.0, 9: 0.4})
        if "ноч" in low or "закат" in low or "дожд" in low or "night_drive" in low:
            return _basis_vec({11: 1.0, 12: 0.5})
        return _basis_vec({13: 1.0})

    monkeypatch.setattr(mixes, "_embed_query", fake_embed)
    forbidden = {ids["remote_only"], ids["broken_local"], ids["disliked_rap"]}

    # 1. «жёсткий русский рэп» must NOT contain Алёна Швец or acoustic rap at the top
    hard = await mixes.prompt_mix(uid, "жёсткий русский рэп", 10)
    hard_ids = [item.candidate.track_id for item in hard]
    assert hard_ids, "must return tracks for hard russian rap"
    assert forbidden.isdisjoint(hard_ids)
    assert ids["sad_pop_shvets"] not in hard_ids[:4]
    assert len(hard_ids) == 10  # a full list beats a short "perfect" one
    assert hard_ids[0] in {ids["hard_rap_oxxy"], ids["hard_rap_obladaet"], ids["hard_rap_pharaoh"]}

    # 2. «акустико-гитарный рэп» ranks Noize MC acoustic rap #1 ahead of drill/trap
    ac_rap = await mixes.prompt_mix(uid, "акустико-гитарный рэп", 10)
    ac_rap_ids = [item.candidate.track_id for item in ac_rap]
    assert forbidden.isdisjoint(ac_rap_ids)
    assert ac_rap_ids[0] == ids["acoustic_rap"]

    # 3. «рэп с джазовыми сэмплами» ranks jazz_rap #1
    j_rap = await mixes.prompt_mix(uid, "рэп с джазовыми сэмплами", 10)
    assert j_rap and j_rap[0].candidate.track_id == ids["jazz_rap"]

    # 4. «спокойный джаз…, без рэпа» keeps jazz-rap, removes pure rap, fills the list
    jazz_clean = await mixes.prompt_mix(uid, "спокойный джаз на вечер, без рэпа", 10)
    jazz_ids = [item.candidate.track_id for item in jazz_clean]
    assert jazz_ids and forbidden.isdisjoint(jazz_ids)
    assert ids["hard_rap_oxxy"] not in jazz_ids
    assert ids["jazz_rap"] in jazz_ids
    assert ids["calm_jazz"] in jazz_ids
    # Hard exclusion wins over list size: 8 = every non-rap track in the fixture.
    assert len(jazz_ids) == 8

    # 5. «спокойная музыка под акустическую гитару» picks Nick Drake ahead of rap
    guitar = await mixes.prompt_mix(uid, "спокойная музыка под акустическую гитару", 10)
    guitar_ids = [item.candidate.track_id for item in guitar]
    assert guitar_ids[0] == ids["calm_guitar"]
    assert ids["hard_rap_obladaet"] not in guitar_ids

    # 6. «ехать по ночной трассе под дождём, на закате» surfaces Nightcall synthwave #1
    drive = await mixes.prompt_mix(uid, "ехать по ночной трассе под дождём, на закате", 10)
    assert drive and drive[0].candidate.track_id == ids["drive_sunset_synth"]

    # 7. Artist exclusion works cleanly
    no_oxxy = await mixes.prompt_mix(uid, "жёсткий русский рэп без Oxxxymiron", 10)
    no_oxxy_ids = [item.candidate.track_id for item in no_oxxy]
    assert ids["hard_rap_oxxy"] not in no_oxxy_ids


@pytest.mark.asyncio
async def test_mood_discovery_personal_and_similar_e2e(rec_catalog, monkeypatch):
    uid, ids = rec_catalog

    async def _no_llm(*_a, **_kw):
        return None

    monkeypatch.setattr(llm, "plan_from_llm", _no_llm)
    llm._plan_cache.clear()
    mixes._qemb_cache.clear()

    async def fake_embed(_text: str) -> list[float] | None:
        return _basis_vec({4: 0.7, 13: 0.7})

    monkeypatch.setattr(mixes, "_embed_query", fake_embed)
    forbidden = {ids["remote_only"], ids["broken_local"], ids["disliked_rap"]}

    # Mood mixes contrast cleanly: party vs focus
    party = await mixes.build(uid, "mood:party", limit=20)
    focus = await mixes.build(uid, "mood:focus", limit=20)
    assert party and focus
    assert forbidden.isdisjoint([x.candidate.track_id for x in party])
    assert forbidden.isdisjoint([x.candidate.track_id for x in focus])
    party_avg_e = sum(x.candidate.energy or 0.0 for x in party) / len(party)
    focus_avg_e = sum(x.candidate.energy or 0.0 for x in focus) / len(focus)
    assert party_avg_e > focus_avg_e + 0.25

    # Discovery: fresh plays are eligible but pushed to the tail (no hard wall);
    # a like without a play is not an anchor; forgotten tracks surface.
    disc = await mixes.build(uid, "discovery", limit=20)
    disc_ids = [x.candidate.track_id for x in disc]
    assert forbidden.isdisjoint(disc_ids)
    assert len(disc_ids) >= 10  # the synthetic catalog is small; a full shelf is
    assert ids["sad_pop_shvets"] not in disc_ids[:8]
    assert ids["party_edm"] not in disc_ids[:8]
    assert ids["old_forgotten_jazz"] in disc_ids
    assert any(
        x.reason_code in ("familiar_anchor", "rediscover", "new_for_you", "liked_unheard", "bridge")
        for x in disc
    )

    # Personal reflects unplayed favorites + onboarding + clean hygiene
    pers = await mixes.build(uid, "personal", limit=20)
    pers_ids = [x.candidate.track_id for x in pers]
    assert pers_ids and forbidden.isdisjoint(pers_ids)

    # Similar / radio for Oxxxymiron excludes seed & forbidden, returns rap neighbours first
    sim = await mixes.similar(uid, ids["hard_rap_oxxy"], limit=5)
    sim_ids = [x.candidate.track_id for x in sim]
    assert sim_ids and ids["hard_rap_oxxy"] not in sim_ids
    assert forbidden.isdisjoint(sim_ids)
    assert sim_ids[0] in {ids["hard_rap_obladaet"], ids["hard_rap_pharaoh"]}

    # Heuristic tagger + eval benchmark suite run cleanly against the populated catalog
    tagged = tagger.heuristic_tags(
        {
            "title": "Acoustic Unplugged",
            "artists": ["Test Artist"],
            "genre": "Acoustic Folk",
            "year": 2018,
            "bpm": 86.0,
            "energy": 0.35,
            "dsp_extra": {"acoustic_proxy": 0.88},
        }
    )
    assert "acoustic_guitar" in tagged["textures"] or "acoustic" in tagged["textures"]
    assert tagged["language"] == "en"

    bench = await eval_rec.benchmark_suite(uid)
    assert "prompts" in bench and "moods" in bench
    assert bench["jaccard_hard_vs_aggressive"] >= 0.35


def test_sequencer_prevents_back_to_back_artists():
    items = [
        Scored(candidate=Candidate(track_id=1, artist="Artist A", album="Alb 1", energy=0.8), score=0.95, reason_code="x"),
        Scored(candidate=Candidate(track_id=2, artist="Artist A", album="Alb 1", energy=0.82), score=0.94, reason_code="x"),
        Scored(candidate=Candidate(track_id=3, artist="Artist B", album="Alb 2", energy=0.76), score=0.90, reason_code="x"),
        Scored(candidate=Candidate(track_id=4, artist="Artist C", album="Alb 3", energy=0.60), score=0.88, reason_code="x"),
    ]
    seq = sequence_playlist(items, arc="wave", seed=0)
    assert [x.candidate.artist for x in seq[:3]] != ["Artist A", "Artist A", "Artist B"]
    assert seq[0].candidate.artist != seq[1].candidate.artist
    assert sorted(x.candidate.track_id for x in seq) == [1, 2, 3, 4]
    assert DEFAULT_PROFILE["pref_parts"]["audio"] >= 0.2
