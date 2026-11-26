"""«О треке»: metadata + measured audio quality + sources.

The player had no way to see what the library actually stores for a track
(quality report: «нет возможности посмотреть метаданные трека, качество»).
"""

from __future__ import annotations

import uuid

import pytest

from app import db
from app.services.identity import identity_hash
from app.v1 import catalog_router


@pytest.fixture(autouse=True)
async def _schema():
    await db.apply_schema()
    yield
    await db.close_engine()


async def _mk_user() -> int:
    s = uuid.uuid4().hex[:8]
    rows = await db.execute_returning(
        "INSERT INTO users (username, password_hash, email, role) VALUES (:u, 'x', :e, 'user') RETURNING id",
        {"u": f"ti-{s}", "e": f"ti-{s}@example.com"},
    )
    return rows[0]["id"]


async def _mk_track(title: str, artists: list[str], dur: int) -> int:
    rows = await db.execute_returning(
        "INSERT INTO tracks (title, artists, album, duration_sec, identity_hash, last_seen_at) "
        "VALUES (:t, :a, '', :d, :h, NOW()) RETURNING id",
        {"t": title, "a": artists, "d": dur, "h": identity_hash(title, artists)},
    )
    return rows[0]["id"]


@pytest.mark.asyncio
async def test_track_info_returns_quality_and_sources():
    uid = await _mk_user()
    tid = await _mk_track("song about quality", ["Artist Q"], 200)
    await db.execute(
        "INSERT INTO track_quality "
        "(track_id, file_path, codec, bitrate_kbps, sample_rate, lossless, tier, bytes, duration_sec) "
        "VALUES (:t, '/data/music/x.flac', 'flac', 900, 44100, TRUE, 2, 123456, 200.4)",
        {"t": tid},
    )
    await db.execute(
        "INSERT INTO track_sources (track_id, connector, external_id, quality_tier, available) "
        "VALUES (:t, 'yandex', '777', 85, TRUE), (:t, 'local', 'local::x.flac', 100, TRUE)",
        {"t": tid},
    )

    info = await catalog_router.get_track_info(str(tid), user={"id": uid})

    assert info.track.id == str(tid)
    assert info.track.title == "song about quality"
    assert info.quality is not None
    assert info.quality.codec == "flac"
    assert info.quality.lossless is True
    assert info.quality.bitrate_kbps == 900
    assert info.quality.measured_duration_ms == 200_400
    # local source is listed first, remotes after
    assert [s.connector for s in info.sources] == ["local", "yandex"]
    assert info.sources[1].quality_tier == 85


@pytest.mark.asyncio
async def test_track_info_without_quality_row():
    uid = await _mk_user()
    tid = await _mk_track("plain track", ["Artist Q"], 180)
    info = await catalog_router.get_track_info(str(tid), user={"id": uid})
    assert info.quality is None
    assert info.sources == []
