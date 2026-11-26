"""Auto-enqueue: new local files get DSP+text, new catalog tracks get text."""

from __future__ import annotations

import pytest

from app import db
from app.connectors.base import TrackMeta
from app.services.rec.queue import MODEL_VERSIONS, enqueue_track
from app.services.search_service import upsert_track_metas


@pytest.fixture(autouse=True)
async def _schema():
    await db.apply_schema()
    await db.execute("TRUNCATE tracks CASCADE")
    yield
    await db.close_engine()


async def _track(title: str = "Auto Song") -> int:
    rows = await db.execute_returning(
        "INSERT INTO tracks (title, artists, duration_sec) "
        "VALUES (:t, '{A}', 10) RETURNING id",
        {"t": title},
    )
    return int(rows[0]["id"])


@pytest.mark.asyncio
async def test_enqueue_track_dsp_and_text():
    tid = await _track()
    await enqueue_track(tid, kinds=("dsp", "text"))

    rows = await db.fetch_all(
        "SELECT kind, model_version FROM index_jobs WHERE track_id = :t ORDER BY kind",
        {"t": tid},
    )
    assert {(r["kind"], r["model_version"]) for r in rows} == {
        ("dsp", MODEL_VERSIONS["dsp"]),
        ("text", MODEL_VERSIONS["text"]),
    }


@pytest.mark.asyncio
async def test_force_requeues_done_audio_and_tags_after_dsp_change():
    tid = await _track()
    await enqueue_track(tid, kinds=("tags", "audio"))
    await db.execute("UPDATE index_jobs SET status = 'done' WHERE track_id = :t", {"t": tid})

    await enqueue_track(tid, kinds=("tags", "audio"))
    rows = await db.fetch_all(
        "SELECT status FROM index_jobs WHERE track_id = :t", {"t": tid}
    )
    assert {r["status"] for r in rows} == {"done"}

    await enqueue_track(tid, kinds=("tags", "audio"), force=True)
    rows = await db.fetch_all(
        "SELECT status FROM index_jobs WHERE track_id = :t", {"t": tid}
    )
    assert {r["status"] for r in rows} == {"pending"}


@pytest.mark.asyncio
async def test_catalog_upsert_enqueues_text():
    meta = TrackMeta(
        connector="yandex",
        external_id="auto-text-1",
        title="Auto Song",
        artists=["A"],
        duration_sec=10,
        quality_tier=65,
    )
    ids = await upsert_track_metas([meta])
    assert ids

    rows = await db.fetch_all(
        "SELECT kind FROM index_jobs WHERE track_id = :t",
        {"t": ids[0]},
    )
    assert [r["kind"] for r in rows] == ["text"]
