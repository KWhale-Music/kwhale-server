"""DSP decoder fallback: Essentia -> ffmpeg WAV for m4a/opus containers."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("essentia")

from app import (
    db,  # noqa: E402
    indexer,  # noqa: E402
)
from app.services.rec.queue import MODEL_VERSIONS, enqueue_track  # noqa: E402


class _FakeLoader:
    """Stand-in for essentia.standard.AudioLoader."""

    def __init__(self, filename: str):
        self.filename = filename

    def __call__(self):
        if self.filename.endswith(".wav"):
            return np.zeros((16, 2), dtype=np.float32), 44100.0
        raise RuntimeError("Could not find stream information")


def _fake_ffmpeg(monkeypatch, returncode: int = 0, stderr: str = ""):
    calls: list[list[str]] = []

    def run(argv, **kwargs):  # noqa: ANN001, ANN003
        calls.append(list(argv))
        if returncode == 0:
            with open(argv[-1], "wb") as fh:
                fh.write(b"RIFF-fake-wav")
        return SimpleNamespace(returncode=returncode, stderr=stderr)

    monkeypatch.setattr(indexer.shutil, "which", lambda name: "/usr/bin/ffmpeg")
    monkeypatch.setattr(indexer.subprocess, "run", run)
    return calls


def test_m4a_falls_back_to_ffmpeg(tmp_path, monkeypatch):
    src = tmp_path / "song.m4a"
    src.write_bytes(b"not-really-aac")
    monkeypatch.setattr(indexer.es, "AudioLoader", _FakeLoader)
    calls = _fake_ffmpeg(monkeypatch)

    samples, rate = indexer._load_audio(str(src))

    assert rate == 44100.0
    assert samples.shape == (16, 2)
    assert len(calls) == 1
    # -vn is what removes the mjpeg cover-art stream that trips Essentia
    assert "-vn" in calls[0]
    assert calls[0][-1].endswith(".wav")


def test_ffmpeg_failure_is_reported_as_corrupt(tmp_path, monkeypatch):
    src = tmp_path / "broken.m4a"
    src.write_bytes(b"broken")
    monkeypatch.setattr(indexer.es, "AudioLoader", _FakeLoader)
    _fake_ffmpeg(monkeypatch, returncode=1, stderr="Invalid data found when processing input")

    with pytest.raises(indexer.CorruptAudio, match="Invalid data"):
        indexer._load_audio(str(src))


@pytest.fixture(autouse=True)
async def _schema():
    await db.apply_schema()
    await db.execute("TRUNCATE tracks CASCADE")
    yield
    await db.close_engine()


async def _track(title: str = "DSP Song") -> int:
    rows = await db.execute_returning(
        "INSERT INTO tracks (title, artists, duration_sec) "
        "VALUES (:t, '{A}', 10) RETURNING id",
        {"t": title},
    )
    return int(rows[0]["id"])


@pytest.mark.asyncio
async def test_enqueue_track_resets_failed_job():
    tid = await _track()
    await db.execute(
        """
        INSERT INTO index_jobs (track_id, kind, model_version, status, attempts, last_error)
        VALUES (:t, 'dsp', :v, 'failed', 5, 'old error')
        """,
        {"t": tid, "v": MODEL_VERSIONS["dsp"]},
    )

    await enqueue_track(tid, kinds=("dsp",))

    row = await db.fetch_one(
        "SELECT status, attempts, last_error FROM index_jobs "
        "WHERE track_id = :t AND kind = 'dsp'",
        {"t": tid},
    )
    assert row == {"status": "pending", "attempts": 0, "last_error": None}


@pytest.mark.asyncio
async def test_reset_stale_running_requeues_only_old_jobs():
    tid = await _track("Stale Song")
    await db.execute(
        """
        INSERT INTO index_jobs (track_id, kind, model_version, status, updated_at)
        VALUES (:t, 'dsp', :v, 'running', NOW() - INTERVAL '10 minutes')
        """,
        {"t": tid, "v": MODEL_VERSIONS["dsp"]},
    )
    fresh = await _track("Fresh Song")
    await db.execute(
        """
        INSERT INTO index_jobs (track_id, kind, model_version, status, updated_at)
        VALUES (:t, 'dsp', :v, 'running', NOW())
        """,
        {"t": fresh, "v": MODEL_VERSIONS["dsp"]},
    )

    assert await indexer.reset_stale_running() == 1

    rows = await db.fetch_all(
        "SELECT track_id, status FROM index_jobs WHERE kind = 'dsp' ORDER BY track_id"
    )
    by_track = {r["track_id"]: r["status"] for r in rows}
    assert by_track[tid] == "pending"
    assert by_track[fresh] == "running"


@pytest.mark.asyncio
async def test_enqueue_track_does_not_reset_done_job():
    tid = await _track("Done Song")
    await db.execute(
        """
        INSERT INTO index_jobs (track_id, kind, model_version, status)
        VALUES (:t, 'dsp', :v, 'done')
        """,
        {"t": tid, "v": MODEL_VERSIONS["dsp"]},
    )

    await enqueue_track(tid, kinds=("dsp",))

    row = await db.fetch_one(
        "SELECT status FROM index_jobs WHERE track_id = :t AND kind = 'dsp'",
        {"t": tid},
    )
    assert row["status"] == "done"
