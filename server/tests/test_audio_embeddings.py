"""Audio embeddings (E9): kind=audio queue, dsp17-v1 backend, padded storage."""

from __future__ import annotations

import pytest

from app import db
from app.services.rec import audio_embed
from app.services.rec.features import feature_vector
from app.services.rec.queue import MODEL_VERSIONS, enqueue_track


@pytest.fixture(autouse=True)
async def _schema():
    await db.apply_schema()
    await db.execute("TRUNCATE tracks CASCADE")
    yield
    await db.close_engine()


async def _track(title: str = "Embedding Song") -> int:
    rows = await db.execute_returning(
        "INSERT INTO tracks (title, artists, duration_sec) "
        "VALUES (:t, '{A}', 10) RETURNING id",
        {"t": title},
    )
    return int(rows[0]["id"])


async def _features(track_id: int) -> None:
    await db.execute(
        """
        INSERT INTO track_features
            (track_id, bpm, beat_confidence, loudness_lufs, danceability, energy,
             valence, spectral_centroid, mfcc_mean)
        VALUES (:t, 128, 0.9, -8.5, 0.7, 0.8, 0.6, 3200, '{1.5,-2,3,-4,5}')
        """,
        {"t": track_id},
    )


async def _audio_job(track_id: int) -> dict:
    job = await db.fetch_one(
        "SELECT id, track_id, kind, model_version FROM index_jobs WHERE track_id = :t",
        {"t": track_id},
    )
    assert job is not None
    return job


@pytest.mark.asyncio
async def test_column_width_matches_backend():
    row = await db.fetch_one(
        "SELECT atttypmod FROM pg_attribute"
        " WHERE attrelid = 'track_emb_audio'::regclass AND attname = 'vec'"
    )
    assert row["atttypmod"] == audio_embed.EMBED_DIM


@pytest.mark.asyncio
async def test_dsp_backend_returns_the_feature_vector():
    features = {"bpm": 128.0, "beat_confidence": 0.9, "energy": 0.8, "valence": 0.6}
    vec = await audio_embed.get_audio_backend().embed(1, features, None)
    assert vec == feature_vector(features)


@pytest.mark.asyncio
async def test_enqueue_track_queues_audio_kind():
    tid = await _track()
    await enqueue_track(tid, kinds=("dsp", "text", "audio"))

    rows = await db.fetch_all(
        "SELECT kind, model_version FROM index_jobs WHERE track_id = :t", {"t": tid}
    )
    assert {(r["kind"], r["model_version"]) for r in rows} == {
        ("dsp", MODEL_VERSIONS["dsp"]),
        ("text", MODEL_VERSIONS["text"]),
        ("audio", MODEL_VERSIONS["audio"]),
    }


@pytest.mark.asyncio
async def test_audio_job_stores_a_padded_embedding():
    indexer = pytest.importorskip("app.indexer")  # essentia is an optional extra
    tid = await _track()
    await _features(tid)
    await enqueue_track(tid, kinds=("audio",))

    await indexer.process(await _audio_job(tid))

    row = await db.fetch_one(
        "SELECT model_version, vec::text AS vec FROM track_emb_audio WHERE track_id = :t",
        {"t": tid},
    )
    assert row is not None, "обработанная задача должна дать строку в track_emb_audio"
    assert row["model_version"] == MODEL_VERSIONS["audio"]
    stored = [float(v) for v in row["vec"].strip("[]").split(",")]
    assert len(stored) == audio_embed.EMBED_DIM

    features_row = await db.fetch_one(
        "SELECT * FROM track_features WHERE track_id = :t", {"t": tid}
    )
    expected = feature_vector(features_row)
    assert stored[: len(expected)] == pytest.approx(expected, abs=1e-6)
    assert stored[len(expected) :] == [0.0] * (audio_embed.EMBED_DIM - len(expected))


@pytest.mark.asyncio
async def test_audio_job_defers_until_the_dsp_row_lands():
    indexer = pytest.importorskip("app.indexer")
    tid = await _track()
    await enqueue_track(tid, kinds=("audio",))
    job = await _audio_job(tid)

    with pytest.raises(indexer.FeaturesNotReady):
        await indexer.process(job)

    await _features(tid)  # the dsp job finished in the meantime
    await indexer.process(job)
    stored = await db.fetch_one("SELECT 1 FROM track_emb_audio WHERE track_id = :t", {"t": tid})
    assert stored is not None


@pytest.mark.asyncio
async def test_drain_audio_processes_a_batch():
    indexer = pytest.importorskip("app.indexer")
    ids: list[int] = []
    for index in range(3):
        tid = await _track(f"Batch Song {index}")
        await _features(tid)
        await enqueue_track(tid, kinds=("audio",))
        ids.append(tid)

    assert await indexer.drain_audio() == 3

    rows = await db.fetch_all("SELECT track_id FROM track_emb_audio")
    assert {int(r["track_id"]) for r in rows} == set(ids)


@pytest.mark.asyncio
async def test_new_local_file_queues_dsp_text_and_audio(tmp_path, monkeypatch):
    from app.services import library, providers, quality

    async def _none(*args, **kwargs):  # noqa: ANN002, ANN003
        return None

    tid = await _track("Hook Song")
    monkeypatch.setattr(library, "music_dir", lambda: tmp_path)
    monkeypatch.setattr(library, "write_tags", lambda *a, **k: None)
    monkeypatch.setattr(library, "fetch_cover", _none)
    monkeypatch.setattr(providers, "get_lyrics", _none)
    monkeypatch.setattr(providers, "itunes_cover", _none)
    monkeypatch.setattr(quality, "measure_track", _none)

    staged = tmp_path / "hook.mp3"
    staged.write_bytes(b"fake-mp3")
    await library.finalize(tid, staged, "audio/mpeg")

    rows = await db.fetch_all(
        "SELECT kind, model_version FROM index_jobs WHERE track_id = :t", {"t": tid}
    )
    assert {(r["kind"], r["model_version"]) for r in rows} == {
        ("dsp", MODEL_VERSIONS["dsp"]),
        ("tags", MODEL_VERSIONS["tags"]),
        ("text", MODEL_VERSIONS["text"]),
        ("audio", MODEL_VERSIONS["audio"]),
    }


@pytest.mark.asyncio
async def test_claim_tags_batch_sql_valid():
    indexer = pytest.importorskip("app.indexer")
    tid = await _track("Tag Claim")
    await enqueue_track(tid, kinds=("tags",))
    claimed = await indexer.claim_tags_batch(10)
    assert len(claimed) == 1
    assert claimed[0]["track_id"] == tid
