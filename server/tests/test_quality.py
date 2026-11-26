"""Measured audio quality: probe, aggregation, lists and safe bulk actions.

The tests use the isolated `kwhale_test` database (conftest guard) and generate
tiny real audio files for the probe; ffmpeg-dependent cases are skipped when it
is not installed.
"""

from __future__ import annotations

import asyncio
import shutil
import uuid
import wave
from dataclasses import replace
from pathlib import Path

import pytest

from app import db
from app.config import settings
from app.services import quality


@pytest.fixture(autouse=True)
async def _schema():
    await db.apply_schema()
    # Quality aggregates are global; isolate each test from tracks/sources left
    # by earlier modules in the shared kwhale_test database.
    await db.execute("TRUNCATE tracks CASCADE")
    await db.execute("DELETE FROM events")
    await db.execute("DELETE FROM download_jobs")
    yield
    await db.close_engine()


async def _mk_user() -> int:
    suffix = uuid.uuid4().hex[:8]
    rows = await db.execute_returning(
        "INSERT INTO users (username, password_hash, email, role) "
        "VALUES (:u, 'x', :e, 'user') RETURNING id",
        {"u": f"q-{suffix}", "e": f"q-{suffix}@example.com"},
    )
    return int(rows[0]["id"])


async def _mk_track(
    title: str,
    *,
    bitrate: int | None = None,
    lossless: bool = False,
    tier: int = 0,
    days_old: int = 0,
    file_path: str | None = None,
    permanent: bool = False,
    with_quality: bool = True,
) -> tuple[int, int]:
    rows = await db.execute_returning(
        "INSERT INTO tracks (title, artists, album, duration_sec) "
        "VALUES (:t, ARRAY['Quality Artist'], 'Quality Album', 180) RETURNING id",
        {"t": title},
    )
    track_id = int(rows[0]["id"])
    source_rows = await db.execute_returning(
        """
        INSERT INTO track_sources
            (track_id, connector, external_id, quality_tier, mime, file_path,
             available, permanent, updated_at)
        VALUES (:t, 'local', :e, 100, 'audio/mpeg', :p, TRUE, :perm,
                NOW() - make_interval(days => :days))
        RETURNING id
        """,
        {
            "t": track_id,
            "e": f"local::test-{uuid.uuid4().hex[:10]}",
            "p": file_path or f"/nonexistent/{uuid.uuid4().hex}.mp3",
            "perm": permanent,
            "days": days_old,
        },
    )
    source_id = int(source_rows[0]["id"])
    if with_quality:
        await db.execute(
            """
            INSERT INTO track_quality
                (track_id, local_source_id, file_path, container, codec,
                 bitrate_kbps, sample_rate, bit_depth, channels, lossless, tier,
                 bytes, duration_sec)
            VALUES (:t, :s, :p, 'mp3', 'mp3', :br, 44100, NULL, 2, :lossless,
                    :tier, :bytes, 180)
            """,
            {
                "t": track_id,
                "s": source_id,
                "p": file_path or f"/nonexistent/{uuid.uuid4().hex}.mp3",
                "br": bitrate,
                "lossless": lossless,
                "tier": tier,
                "bytes": (bitrate or 0) * 1024,
            },
        )
    return track_id, source_id


def _write_wav(path: Path, seconds: float = 0.2) -> None:
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(2)
        wav.setsampwidth(2)
        wav.setframerate(44100)
        wav.writeframes(b"\x00\x00\x00\x00" * int(44100 * seconds))


def _ffmpeg(path: Path, codec_args: list[str]) -> None:
    shutil.which("ffmpeg") or pytest.skip("ffmpeg is not installed")
    import subprocess

    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "anullsrc=r=44100:cl=stereo",
            "-t",
            "0.5",
            *codec_args,
            str(path),
        ],
        check=True,
    )


def test_tier_boundaries():
    assert quality.tier_for(None, False) == quality.TIER_UNKNOWN
    assert quality.tier_for(125, False) == quality.TIER_LOW
    assert quality.tier_for(126, False) == quality.TIER_128
    assert quality.tier_for(189, False) == quality.TIER_128
    assert quality.tier_for(190, False) == quality.TIER_192
    assert quality.tier_for(253, False) == quality.TIER_192
    assert quality.tier_for(254, False) == quality.TIER_256
    assert quality.tier_for(317, False) == quality.TIER_256
    assert quality.tier_for(318, False) == quality.TIER_320
    assert quality.tier_for(1000, True) == quality.TIER_LOSSLESS


def test_probe_wav_is_lossless(tmp_path):
    path = tmp_path / "tone.wav"
    _write_wav(path)
    result = quality.probe_file(path)
    assert result.lossless is True
    assert result.tier == quality.TIER_LOSSLESS
    assert result.bit_depth == 16
    assert result.sample_rate == 44100
    assert result.bitrate_kbps == 1411


def test_probe_mp3_and_flac(tmp_path):
    mp3 = tmp_path / "tone.mp3"
    _ffmpeg(mp3, ["-c:a", "libmp3lame", "-b:a", "128k"])
    result = quality.probe_file(mp3)
    assert result.lossless is False
    assert 110 <= (result.bitrate_kbps or 0) <= 150
    assert result.tier == quality.TIER_128

    flac = tmp_path / "tone.flac"
    _ffmpeg(flac, ["-c:a", "flac"])
    result = quality.probe_file(flac)
    assert result.lossless is True
    assert result.tier == quality.TIER_LOSSLESS


async def test_scan_creates_quality_row(tmp_path):
    path = tmp_path / "scan.wav"
    _write_wav(path)
    track_id, _ = await _mk_track(
        "Scan me",
        file_path=str(path),
        with_quality=False,
    )
    result = await quality.scan(force=True)
    assert result["scanned"] >= 1
    row = await db.fetch_one(
        "SELECT tier, lossless, codec FROM track_quality WHERE track_id = :t",
        {"t": track_id},
    )
    assert row is not None
    assert row["tier"] == quality.TIER_LOSSLESS
    assert row["lossless"] is True
    assert row["codec"] == "pcm"


async def test_overview_counts_library_downloads_and_plays():
    await _mk_track(
        "Q lossless", bitrate=1411, lossless=True, tier=6, days_old=1
    )
    await _mk_track("Q low", bitrate=128, tier=2, days_old=2)
    await _mk_track("Q old 320", bitrate=320, tier=5, days_old=40)
    played_track, _ = await _mk_track("Q played", bitrate=256, tier=4, days_old=3)
    user_id = await _mk_user()
    await db.execute(
        """
        INSERT INTO events (id, user_id, type, track_id, occurred_at)
        VALUES (:id, :u, 'play', :t, NOW())
        """,
        {"id": uuid.uuid4(), "u": user_id, "t": played_track},
    )

    data = await quality.overview(30)
    lib = data["library"]
    assert lib["tracks"] == 4
    assert lib["measured"] == 4
    assert lib["lossless"] == 1
    assert lib["below192"] == 1
    assert lib["distribution"]["lossless"] == 1
    assert lib["distribution"]["b128"] == 1
    assert lib["distribution"]["gte320"] == 1
    assert lib["distribution"]["b256"] == 1
    assert lib["distribution"]["unknown"] == 0
    assert abs((lib["avgLossyBitrateKbps"] or 0) - 234.7) < 0.5

    downloaded = data["downloaded"]
    assert downloaded["tracks"] == 3  # 40-day-old 320 is outside the window
    assert downloaded["lossless"] == 1

    all_time = await quality.overview(0)
    assert all_time["downloaded"]["tracks"] == 4  # days=0 means all time

    played = data["played"]
    assert played["tracks"] == 1
    assert played["measured"] == 1
    assert played["estimated"] == 0
    assert played["lossless"] == 0
    assert played["avgLossyBitrateKbps"] == 256.0


async def test_list_filters_and_upgrade_flag():
    low, low_source = await _mk_track("List low", bitrate=128, tier=2)
    await _mk_track("List flac", bitrate=1411, lossless=True, tier=6)
    unmeasured, _ = await _mk_track("List unknown", with_quality=False)
    await db.execute(
        """
        INSERT INTO track_sources (track_id, connector, external_id, quality_tier, available)
        VALUES (:t, 'yandex', :e, 65, TRUE)
        """,
        {"t": low, "e": f"yandex::{uuid.uuid4().hex}"},
    )

    low_page = await quality.list_tracks(max_bitrate=191)
    assert low_page["total"] == 1
    assert low_page["items"][0]["title"] == "List low"
    assert low_page["items"][0]["canUpgrade"] is True
    assert low_page["items"][0]["best"]["label"].startswith("MP3 320")

    unknown_page = await quality.list_tracks(tier=0)
    assert unknown_page["total"] == 1
    assert unknown_page["items"][0]["id"] == str(unmeasured)
    assert unknown_page["items"][0]["quality"] is None

    worst = await quality.list_tracks(sort="worst")
    assert worst["total"] == 3
    # known worst first, unknown last
    assert worst["items"][0]["title"] == "List low"
    assert worst["items"][-1]["id"] == str(unmeasured)
    assert low_source > 0


async def test_delete_moves_to_orphaned_and_protects_uploads(tmp_path, monkeypatch):
    monkeypatch.setattr(quality, "settings", replace(settings, music_dir=tmp_path))
    track_file = tmp_path / "Artist" / "song.mp3"
    track_file.parent.mkdir(parents=True)
    track_file.write_bytes(b"x" * 2048)
    track_id, _ = await _mk_track("Delete me", file_path=str(track_file), bitrate=128)

    kept_file = tmp_path / "Artist" / "upload.mp3"
    kept_file.write_bytes(b"y" * 1024)
    permanent_id, _ = await _mk_track(
        "Upload keep", file_path=str(kept_file), bitrate=320, permanent=True
    )

    result = await quality.delete_local([track_id, permanent_id])
    assert result["deletedFiles"] == 1
    assert result["bytesFreed"] == 2048
    assert not track_file.exists()
    orphaned = list((tmp_path / "_orphaned" / "quality").rglob("deleted-*"))
    assert len(orphaned) == 1
    row = await db.fetch_one(
        "SELECT 1 FROM track_sources WHERE track_id = :t AND connector = 'local'",
        {"t": track_id},
    )
    assert row is None
    qrow = await db.fetch_one(
        "SELECT 1 FROM track_quality WHERE track_id = :t", {"t": track_id}
    )
    assert qrow is None
    assert kept_file.exists()
    assert any(item["trackId"] == str(permanent_id) for item in result["skipped"])


async def test_redownload_force_queues_and_skips_permanent():
    track_id, _ = await _mk_track("Redownload me", bitrate=128)
    result = await quality.redownload([track_id])
    assert result["queued"] == 1
    job = await db.fetch_one(
        "SELECT replace_local, status FROM download_jobs WHERE track_id = :t",
        {"t": track_id},
    )
    assert job is not None
    assert job["replace_local"] is True
    assert job["status"] == "pending"

    permanent_id, _ = await _mk_track("No replace upload", bitrate=320, permanent=True)
    result = await quality.redownload([permanent_id])
    assert result["queued"] == 0
    assert result["skipped"]
    job = await db.fetch_one(
        "SELECT 1 FROM download_jobs WHERE track_id = :t", {"t": permanent_id}
    )
    assert job is None


async def test_start_scan_is_running_immediately():
    started = await quality.start_scan(limit=1)
    assert started is True
    assert quality.scan_status()["running"] is True
    for _ in range(100):
        if not quality.scan_status()["running"]:
            break
        await asyncio.sleep(0.02)
    assert quality.scan_status()["running"] is False
    assert quality.scan_status()["finishedAt"] is not None


async def test_quarantine_old_copies_keeps_new_file(tmp_path, monkeypatch):
    monkeypatch.setattr(quality, "settings", replace(settings, music_dir=tmp_path))
    track_id, _ = await _mk_track("Replace me", bitrate=128)
    old_dir = tmp_path / "old"
    old_dir.mkdir()
    old_file = old_dir / "old.mp3"
    old_file.write_bytes(b"old")
    await db.execute(
        "UPDATE track_sources SET file_path = :p WHERE track_id = :t AND connector = 'local'",
        {"p": str(old_file), "t": track_id},
    )
    new_file = tmp_path / "new.mp3"
    new_file.write_bytes(b"new")
    await db.execute(
        """
        INSERT INTO track_sources
            (track_id, connector, external_id, quality_tier, mime, file_path, available)
        VALUES (:t, 'local', :e, 100, 'audio/mpeg', :p, TRUE)
        """,
        {"t": track_id, "e": f"local::new-{uuid.uuid4().hex}", "p": str(new_file)},
    )

    result = await quality.quarantine_old_copies(track_id, new_file)
    assert result["moved"] == 1
    assert not old_file.exists()
    rows = await db.fetch_all(
        "SELECT file_path FROM track_sources WHERE track_id = :t AND connector = 'local'",
        {"t": track_id},
    )
    assert [row["file_path"] for row in rows] == [str(new_file)]


def test_probe_falls_back_to_ffprobe_for_unreadable_containers(monkeypatch, tmp_path):
    shutil.which("ffprobe") or pytest.skip("ffprobe is not installed")
    path = tmp_path / "fragmented.m4a"
    _write_wav(path)
    # Simulate mutagen returning nothing, as it does for fragmented MP4/HLS.
    monkeypatch.setattr(quality, "MutagenFile", lambda _path: None)

    result = quality.probe_file(path)

    assert result.container == "m4a"
    assert result.bitrate_kbps and result.bitrate_kbps > 0
    assert result.sample_rate == 44100
