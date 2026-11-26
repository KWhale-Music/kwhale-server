"""Admin quality API: auth, overview/list, scan and safe bulk actions."""

from __future__ import annotations

import uuid
import wave
from dataclasses import replace
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from app import db
from app.auth import ensure_seed_user, hash_password
from app.config import settings
from app.main import app
from app.services import quality


@pytest.fixture(autouse=True)
async def _schema():
    await db.apply_schema()
    await db.execute("TRUNCATE tracks CASCADE")
    await db.execute("DELETE FROM events")
    await db.execute("DELETE FROM download_jobs")
    await ensure_seed_user()
    yield
    await db.close_engine()


@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


async def _login(client: AsyncClient, email: str, password: str) -> dict[str, str]:
    response = await client.post(
        "/v1/auth/login", json={"email": email, "password": password}
    )
    assert response.status_code == 200, response.text
    return {"Authorization": f"Bearer {response.json()['accessToken']}"}


async def _admin(client: AsyncClient) -> dict[str, str]:
    return await _login(client, settings.seed_email, settings.seed_password)


async def _user(client: AsyncClient) -> dict[str, str]:
    suffix = uuid.uuid4().hex[:8]
    email = f"qadmin-{suffix}@example.com"
    await db.execute(
        "INSERT INTO users (username, password_hash, email, role) "
        "VALUES (:u, :p, :e, 'user')",
        {"u": f"qadmin-{suffix}", "p": hash_password("pw"), "e": email},
    )
    return await _login(client, email, "pw")


async def _track(file_path: str, *, bitrate: int = 128, days_old: int = 0) -> int:
    rows = await db.execute_returning(
        "INSERT INTO tracks (title, artists, album, duration_sec) "
        "VALUES (:t, ARRAY['Admin Artist'], 'Admin Album', 180) RETURNING id",
        {"t": f"Admin Q {uuid.uuid4().hex[:8]}"},
    )
    track_id = int(rows[0]["id"])
    await db.execute(
        """
        INSERT INTO track_sources
            (track_id, connector, external_id, quality_tier, mime, file_path,
             available, updated_at)
        VALUES (:t, 'local', :e, 100, 'audio/mpeg', :p, TRUE,
                NOW() - make_interval(days => :d))
        """,
        {
            "t": track_id,
            "e": f"local::admin-{uuid.uuid4().hex[:8]}",
            "p": file_path,
            "d": days_old,
        },
    )
    await db.execute(
        """
        INSERT INTO track_quality
            (track_id, local_source_id, file_path, codec, bitrate_kbps, tier,
             bytes, lossless, sample_rate, channels)
        SELECT :t, id, :p, 'mp3', :b, :tier, 1024, FALSE, 44100, 2
        FROM track_sources WHERE track_id = :t AND connector = 'local'
        LIMIT 1
        """,
        {
            "t": track_id,
            "p": file_path,
            "b": bitrate,
            "tier": quality.tier_for(bitrate, False),
        },
    )
    return track_id


def _write_wav(path: Path) -> None:
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(2)
        wav.setsampwidth(2)
        wav.setframerate(44100)
        wav.writeframes(b"\x00\x00\x00\x00" * 4410)


async def test_quality_overview_requires_admin(client):
    headers = await _user(client)
    response = await client.get("/admin/api/quality/overview", headers=headers)
    assert response.status_code == 403


async def test_overview_and_tracks(client, tmp_path):
    await _track(str(tmp_path / "a.mp3"), bitrate=128, days_old=1)
    await _track(str(tmp_path / "b.mp3"), bitrate=320, days_old=40)
    headers = await _admin(client)

    overview = await client.get("/admin/api/quality/overview?days=30", headers=headers)
    assert overview.status_code == 200
    data = overview.json()
    assert data["library"]["tracks"] == 2
    assert data["library"]["below192"] == 1
    assert data["downloaded"]["tracks"] == 1

    page = await client.get(
        "/admin/api/quality/tracks?days=30&maxBitrate=191", headers=headers
    )
    assert page.status_code == 200
    body = page.json()
    assert body["total"] == 1
    assert body["items"][0]["quality"]["bitrateKbps"] == 128
    assert body["items"][0]["canUpgrade"] is False  # no other source


async def test_rescan_selected_track(client, tmp_path):
    path = tmp_path / "scan.wav"
    _write_wav(path)
    track_id = await _track(str(path))
    # Remove the seeded quality row: this test verifies the measurement path.
    await db.execute("DELETE FROM track_quality WHERE track_id = :t", {"t": track_id})
    headers = await _admin(client)
    response = await client.post(
        "/admin/api/quality/rescan",
        json={"trackIds": [track_id]},
        headers=headers,
    )
    assert response.status_code == 200
    body = response.json()
    assert body["mode"] == "selected"
    assert body["scanned"] == 1
    row = await db.fetch_one(
        "SELECT tier, lossless FROM track_quality WHERE track_id = :t", {"t": track_id}
    )
    assert row is not None
    assert row["tier"] == 6
    assert row["lossless"] is True


async def test_redownload_and_delete_actions(client, tmp_path, monkeypatch):
    monkeypatch.setattr(quality, "settings", replace(settings, music_dir=tmp_path))
    path = tmp_path / "delete-me.mp3"
    path.write_bytes(b"x" * 100)
    track_id = await _track(str(path))
    headers = await _admin(client)

    redownload = await client.post(
        "/admin/api/quality/redownload",
        json={"trackIds": [track_id]},
        headers=headers,
    )
    assert redownload.status_code == 200
    assert redownload.json()["queued"] == 1
    job = await db.fetch_one(
        "SELECT replace_local FROM download_jobs WHERE track_id = :t", {"t": track_id}
    )
    assert job is not None
    assert job["replace_local"] is True

    deleted = await client.post(
        "/admin/api/quality/delete",
        json={"trackIds": [track_id]},
        headers=headers,
    )
    assert deleted.status_code == 200
    assert deleted.json()["deletedFiles"] == 1
    assert not path.exists()
    assert list((tmp_path / "_orphaned" / "quality").rglob("deleted-*"))


async def test_scan_status_endpoint(client):
    headers = await _admin(client)
    response = await client.get("/admin/api/quality/scan", headers=headers)
    assert response.status_code == 200
    assert "running" in response.json()
