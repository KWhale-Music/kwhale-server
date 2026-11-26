"""Downloaded content must be the recording the track row claims.

Regression for the «body double»/«amy» mixup (2026-11-18): a mislabeled
upstream source (catalog said «body double», the file was «amy», 167 s vs
112 s) passed the only existing check — readability, and only on forced
re-downloads — and entered the managed library under the right tags.
"""

from __future__ import annotations

import pytest

from app import db
from app.repair import content_audit
from app.services.downloader import content_mismatch


@pytest.fixture(autouse=True)
async def _schema():
    await db.apply_schema()
    yield
    await db.close_engine()


async def _mk_track(title: str, dur: int | None) -> int:
    rows = await db.execute_returning(
        "INSERT INTO tracks (title, artists, album, duration_sec, last_seen_at) "
        "VALUES (:t, '{A}', '', :d, NOW()) RETURNING id",
        {"t": title, "d": dur},
    )
    return rows[0]["id"]


async def _measured(track_id: int, dur: float, path: str = "/tmp/x.mp3") -> None:
    await db.execute(
        "INSERT INTO track_quality (track_id, file_path, duration_sec) "
        "VALUES (:i, :p, :d)",
        {"i": track_id, "p": path, "d": dur},
    )


@pytest.mark.asyncio
async def test_wrong_recording_is_rejected():
    tid = await _mk_track("body double", 112)
    reason = await content_mismatch(tid, 167.1)
    assert reason is not None and "content mismatch" in reason


@pytest.mark.asyncio
async def test_close_enough_duration_is_accepted():
    tid = await _mk_track("body double", 112)
    assert await content_mismatch(tid, 125.0) is None


@pytest.mark.asyncio
async def test_preview_placeholder_row_cannot_judge():
    # canonical 30 s is a junk preview duration — the row is the suspect
    tid = await _mk_track("preview junk", 30)
    assert await content_mismatch(tid, 215.0) is None


@pytest.mark.asyncio
async def test_unknown_durations_pass():
    tid = await _mk_track("no duration", None)
    assert await content_mismatch(tid, 200.0) is None
    tid2 = await _mk_track("unmeasured", 112)
    assert await content_mismatch(tid2, None) is None


@pytest.mark.asyncio
async def test_content_audit_splits_and_requeues():
    bad = await _mk_track("bad file", 112)
    await _measured(bad, 167.1)
    junk = await _mk_track("junk canonical", 30)
    await _measured(junk, 215.0)
    good = await _mk_track("good file", 112)
    await _measured(good, 113.0)

    report = await content_audit(fix=False, dry_run=True)
    bad_ids = {r["track_id"] for r in report["wrong_files"]}
    junk_ids = {r["track_id"] for r in report["suspect_rows"]}
    assert bad in bad_ids and good not in bad_ids
    assert junk in junk_ids

    fixed = await content_audit(fix=True)
    requeued = {r["track_id"] for r in fixed["requeued"]}
    assert requeued == {bad}, "на перекачку встаёт только трек с чужим файлом"
    job = await db.fetch_one(
        "SELECT status, replace_local FROM download_jobs WHERE track_id = :i", {"i": bad}
    )
    assert job is not None and job["status"] == "pending" and job["replace_local"] is True
