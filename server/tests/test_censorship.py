"""Censorship report endpoint: POST/DELETE /v1/tracks/{id}/censorship.

The Android client already ships «Отправить на цензуру» (TrackMenuSheet +
player) against this contract: POST {reason, comment, positionMs} ->
{ok, reportId, status} with Idempotency-Key; DELETE cancels; Track carries
isCensorshipReported. Before the fix the client failed with «На сервере ещё не
подключён эндпоинт цензуры».
"""

import uuid

import pytest
from httpx import ASGITransport, AsyncClient

from app import db
from app.auth import ensure_seed_user, hash_password
from app.config import settings
from app.main import app


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
    email = f"censor-{s}@example.com"
    await db.execute(
        "INSERT INTO users (username, password_hash, email, role) VALUES (:u, :p, :e, 'user')",
        {"u": f"censor-{s}", "p": hash_password("pw"), "e": email},
    )
    return await _login(client, email, "pw")


async def _mk_track() -> int:
    rows = await db.execute_returning(
        "INSERT INTO tracks (title, artists, album, duration_sec, last_seen_at) "
        "VALUES ('Censored Song', ARRAY['Adamant'], '', 120, NOW()) RETURNING id"
    )
    return rows[0]["id"]


@pytest.mark.asyncio
async def test_report_censorship_and_flag_in_track_dto(client):
    tid = await _mk_track()
    headers = await _user_headers(client)

    r = await client.post(
        f"/v1/tracks/{tid}/censorship",
        json={"reason": "censored", "comment": "слова заменены звёздочками", "positionMs": 42000},
        headers={**headers, "Idempotency-Key": str(uuid.uuid4())},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ok"] is True
    assert body["status"] == "pending"
    assert body["reportId"]

    # the flag must round-trip on the track DTO
    t = await client.get(f"/v1/tracks/{tid}", headers=headers)
    assert t.status_code == 200
    assert t.json()["isCensorshipReported"] is True

    # and be absent for tracks without a report
    other = await _mk_track()
    t2 = await client.get(f"/v1/tracks/{other}", headers=headers)
    assert t2.json()["isCensorshipReported"] is False

    await db.execute("DELETE FROM censorship_reports WHERE track_id = :t", {"t": tid})
    await db.execute("DELETE FROM tracks WHERE id IN (:t, :o)", {"t": tid, "o": other})


@pytest.mark.asyncio
async def test_report_is_idempotent_per_user_track(client):
    tid = await _mk_track()
    headers = await _user_headers(client)
    key = str(uuid.uuid4())

    r1 = await client.post(
        f"/v1/tracks/{tid}/censorship",
        json={"reason": "censored"},
        headers={**headers, "Idempotency-Key": key},
    )
    r2 = await client.post(
        f"/v1/tracks/{tid}/censorship",
        json={"reason": "censored"},
        headers={**headers, "Idempotency-Key": key},
    )
    assert r1.status_code == 200 and r2.status_code == 200
    assert r1.json()["reportId"] == r2.json()["reportId"], "повтор не должен плодить отчёты"

    n = await db.fetch_one(
        "SELECT COUNT(*) AS n FROM censorship_reports WHERE track_id = :t", {"t": tid}
    )
    assert int(n["n"]) == 1

    await db.execute("DELETE FROM censorship_reports WHERE track_id = :t", {"t": tid})
    await db.execute("DELETE FROM tracks WHERE id = :t", {"t": tid})


@pytest.mark.asyncio
async def test_cancel_censorship_report(client):
    tid = await _mk_track()
    headers = await _user_headers(client)
    r = await client.post(
        f"/v1/tracks/{tid}/censorship",
        json={"reason": "censored"},
        headers={**headers, "Idempotency-Key": str(uuid.uuid4())},
    )
    assert r.status_code == 200

    d = await client.delete(f"/v1/tracks/{tid}/censorship", headers=headers)
    assert d.status_code == 204

    n = await db.fetch_one(
        "SELECT COUNT(*) AS n FROM censorship_reports WHERE track_id = :t", {"t": tid}
    )
    assert int(n["n"]) == 0, "DELETE должен снимать отчёт"
    t = await client.get(f"/v1/tracks/{tid}", headers=headers)
    assert t.json()["isCensorshipReported"] is False

    await db.execute("DELETE FROM tracks WHERE id = :t", {"t": tid})


@pytest.mark.asyncio
async def test_censorship_requires_auth(client):
    tid = await _mk_track()
    r = await client.post(f"/v1/tracks/{tid}/censorship", json={"reason": "censored"})
    assert r.status_code in (401, 403)
    d = await client.delete(f"/v1/tracks/{tid}/censorship")
    assert d.status_code in (401, 403)
    await db.execute("DELETE FROM tracks WHERE id = :t", {"t": tid})


@pytest.mark.asyncio
async def test_admin_lists_and_resolves_reports(client):
    tid = await _mk_track()
    headers = await _user_headers(client)
    r = await client.post(
        f"/v1/tracks/{tid}/censorship",
        json={"reason": "censored", "comment": "check"},
        headers={**headers, "Idempotency-Key": str(uuid.uuid4())},
    )
    report_id = r.json()["reportId"]

    admin = await _login(client, settings.seed_email, settings.seed_password)
    listing = await client.get("/admin/api/censorship", headers=admin)
    assert listing.status_code == 200
    items = listing.json()["items"]
    mine = [i for i in items if i["trackId"] == tid]
    assert mine, "админ должен видеть жалобу"
    assert mine[0]["comment"] == "check"

    res = await client.post(
        f"/admin/api/censorship/{report_id}/resolve",
        json={"status": "resolved"},
        headers=admin,
    )
    assert res.status_code == 200
    assert res.json()["status"] == "resolved"

    await db.execute("DELETE FROM censorship_reports WHERE track_id = :t", {"t": tid})
    await db.execute("DELETE FROM tracks WHERE id = :t", {"t": tid})
