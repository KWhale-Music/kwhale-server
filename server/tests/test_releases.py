"""In-app updater: release endpoint + parity upload paths.

Contract (client side is already specced): GET /v1/app-release/latest returns
{versionCode, versionName, notes, sha256, sizeBytes, downloadUrl}; the APK is
served from the same origin at /v1/app-release/apk. Admin upload and the
manual inbox drop feed the same table.
"""

import json
import uuid

import pytest
from httpx import ASGITransport, AsyncClient

from app import db
from app.auth import ensure_seed_user, hash_password
from app.config import settings
from app.main import app
from app.services import releases


@pytest.fixture(autouse=True)
async def _schema():
    await db.apply_schema()
    await ensure_seed_user()
    yield
    await db.close_engine()


@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


async def _login(client, email, password):
    r = await client.post("/v1/auth/login", json={"email": email, "password": password})
    assert r.status_code == 200, r.text
    return {"Authorization": f"Bearer {r.json()['accessToken']}"}


async def _user_headers(client):
    s = uuid.uuid4().hex[:8]
    email = f"upd-{s}@example.com"
    await db.execute(
        "INSERT INTO users (username, password_hash, email, role) VALUES (:u, :p, :e, 'user')",
        {"u": f"upd-{s}", "p": hash_password("pw"), "e": email},
    )
    return await _login(client, email, "pw")


async def _cleanup_releases():
    from pathlib import Path

    rows = await db.fetch_all("SELECT file_path FROM app_releases")
    for r in rows:
        p = r.get("file_path") or ""
        if p:
            Path(p).unlink(missing_ok=True)
    await db.execute("DELETE FROM app_releases")


@pytest.mark.asyncio
async def test_latest_404_when_nothing_published(client):
    headers = await _user_headers(client)
    r = await client.get("/v1/app-release/latest", headers=headers)
    assert r.status_code == 404
    await _cleanup_releases()


@pytest.mark.asyncio
async def test_admin_upload_and_latest_roundtrip(client):
    admin = await _login(client, settings.seed_email, settings.seed_password)
    apk = b"PK\x03\x04 fake apk bytes"
    r = await client.post(
        "/admin/api/app-releases?versionCode=25&versionName=0.1.15&notes=fixes",
        content=apk,
        headers=admin,
    )
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["versionCode"] == 25
    assert body["sha256"]

    user = await _user_headers(client)
    latest = await client.get("/v1/app-release/latest", headers=user)
    assert latest.status_code == 200
    rel = latest.json()
    assert rel["versionCode"] == 25
    assert rel["versionName"] == "0.1.15"
    assert rel["downloadUrl"] == "/v1/app-release/apk"

    dl = await client.get("/v1/app-release/apk", headers=user)
    assert dl.status_code == 200
    assert dl.content == apk

    await _cleanup_releases()


@pytest.mark.asyncio
async def test_inbox_manual_parity(client):
    """scp-style drop: inbox/client.json + client.apk — same endpoint serves it."""
    inbox = releases.inbox_dir()
    apk_path = inbox / "client.apk"
    meta_path = inbox / "client.json"
    apk_path.write_bytes(b"fake manual apk")
    meta_path.write_text(
        json.dumps({"versionCode": 30, "versionName": "0.2.0", "file": "client.apk"})
    )
    try:
        user = await _user_headers(client)
        latest = await client.get("/v1/app-release/latest", headers=user)
        assert latest.status_code == 200
        rel = latest.json()
        assert rel["versionCode"] == 30, "inbox-релиз должен участвовать наравне с upload"
        dl = await client.get("/v1/app-release/apk", headers=user)
        assert dl.content == b"fake manual apk"
    finally:
        apk_path.unlink(missing_ok=True)
        meta_path.unlink(missing_ok=True)
        await _cleanup_releases()


@pytest.mark.asyncio
async def test_upload_requires_admin(client):
    user = await _user_headers(client)
    r = await client.post(
        "/admin/api/app-releases?versionCode=1&versionName=x", content=b"apk", headers=user
    )
    assert r.status_code in (401, 403)
    await _cleanup_releases()
