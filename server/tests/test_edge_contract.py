"""HTTP/CDN edge-readiness contract tests (HEAD, Range, inline, HMAC media URLs, Vary/no-store, DASH order)."""

from __future__ import annotations

import asyncio
import time

import pytest
from httpx import ASGITransport, AsyncClient

from app import auth, db
from app.config import settings
from app.connectors import dash
from app.connectors.base import StreamInfo
from app.main import app
from app.services import stream_service


@pytest.fixture(autouse=True)
async def _schema():
    auth.clear_user_cache()
    await db.apply_schema()
    await auth.ensure_seed_user()
    yield
    auth.clear_user_cache()
    await db.close_engine()


@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest.fixture
async def admin_headers(client):
    r = await client.post(
        "/v1/auth/login",
        json={"email": settings.seed_email, "password": settings.seed_password},
    )
    assert r.status_code == 200
    return {"Authorization": f"Bearer {r.json()['accessToken']}"}


async def _mk_local_track(tmp_path, content: bytes = b"0123456789") -> int:
    rows = await db.execute_returning(
        """
        INSERT INTO tracks (title, artists, album, duration_sec, last_seen_at)
        VALUES ('Edge Track', '{Artist}', 'Album', 120, NOW())
        RETURNING id
        """
    )
    tid = rows[0]["id"]
    f = tmp_path / f"track-{tid}.mp3"
    f.write_bytes(content)
    await db.execute(
        """
        INSERT INTO track_sources (track_id, connector, external_id, quality_tier, mime, file_path, available)
        VALUES (:t, 'local', :e, 100, 'audio/mpeg', :p, TRUE)
        """,
        {"t": tid, "e": f"local::edge-{tid}.mp3", "p": str(f)},
    )
    return tid


@pytest.mark.asyncio
async def test_stream_head_range_and_inline_disposition(client, admin_headers, tmp_path):
    tid = await _mk_local_track(tmp_path, b"abcdefghij")

    head_full = await client.head(f"/v1/tracks/{tid}/stream", headers=admin_headers)
    assert head_full.status_code == 200
    assert head_full.content == b""
    assert head_full.headers["content-length"] == "10"
    assert head_full.headers["accept-ranges"] == "bytes"
    assert "attachment" not in head_full.headers.get("content-disposition", "").lower()

    head_part = await client.head(
        f"/v1/tracks/{tid}/stream",
        headers={**admin_headers, "Range": "bytes=2-5"},
    )
    assert head_part.status_code == 206
    assert head_part.content == b""
    assert head_part.headers["content-range"] == "bytes 2-5/10"
    assert head_part.headers["content-length"] == "4"

    head_bad = await client.head(
        f"/v1/tracks/{tid}/stream",
        headers={**admin_headers, "Range": "bytes=50-60"},
    )
    assert head_bad.status_code == 416

    get_full = await client.get(f"/v1/tracks/{tid}/stream", headers=admin_headers)
    assert get_full.status_code == 200
    assert get_full.content == b"abcdefghij"
    assert "attachment" not in get_full.headers.get("content-disposition", "").lower()


@pytest.mark.asyncio
async def test_signed_media_url_caching_and_tamper_rejection(client, tmp_path):
    tid = await _mk_local_track(tmp_path, b"0123456789")
    path = f"/v1/tracks/{tid}/stream"
    exp, sig = auth.sign_media_path(path, ttl_sec=3600)

    r_ok = await client.get(f"{path}?exp={exp}&sig={sig}", headers={"Range": "bytes=0-4"})
    assert r_ok.status_code == 206
    assert r_ok.content == b"01234"
    assert "public" in r_ok.headers.get("cache-control", "").lower()

    r_bad_sig = await client.get(f"{path}?exp={exp}&sig=0000")
    assert r_bad_sig.status_code == 401

    r_wrong_track = await client.get(f"/v1/tracks/{tid + 1}/stream?exp={exp}&sig={sig}")
    assert r_wrong_track.status_code == 401

    r_expired = await client.get(
        f"{path}?exp={int(time.time()) - 10}&sig={sig}"
    )
    assert r_expired.status_code == 401


@pytest.mark.asyncio
async def test_private_json_cache_hygiene_vs_public_images(client, admin_headers, tmp_path):
    tid = await _mk_local_track(tmp_path)
    meta = await client.get(f"/v1/tracks/{tid}", headers=admin_headers)
    assert meta.status_code == 200
    cc = meta.headers.get("cache-control", "").lower()
    assert "private" in cc and "no-store" in cc
    assert "authorization" in meta.headers.get("vary", "").lower()

    img = await client.get("/v1/images/placeholder?size=xs")
    assert img.status_code == 200
    img_cc = img.headers.get("cache-control", "").lower()
    assert "public" in img_cc and "immutable" in img_cc


@pytest.mark.asyncio
async def test_current_user_ttl_cache(monkeypatch, client, admin_headers):
    auth.clear_user_cache()
    orig_fetch_one = db.fetch_one
    calls = 0

    async def counting_fetch_one(sql: str, params=None):
        nonlocal calls
        if "FROM users WHERE id" in sql:
            calls += 1
        return await orig_fetch_one(sql, params)

    monkeypatch.setattr(db, "fetch_one", counting_fetch_one)
    r1 = await client.get("/v1/me", headers=admin_headers)
    r2 = await client.get("/v1/me", headers=admin_headers)
    assert r1.status_code == 200 and r2.status_code == 200
    assert calls == 1


@pytest.mark.asyncio
async def test_proxy_head_falls_back_from_405_to_get_headers(monkeypatch):
    class FakeGetResp:
        status_code = 206
        headers = {
            "content-range": "bytes 0-99/1000",
            "content-length": "100",
            "accept-ranges": "bytes",
            "content-type": "audio/mp4",
        }

    class FakeHeadResp:
        status_code = 405
        headers: dict[str, str] = {}

    class FakeClient:
        def __init__(self, *a, **kw):
            pass

        async def head(self, url, headers=None):
            return FakeHeadResp()

        def stream(self, method, url, headers=None):
            class _Ctx:
                async def __aenter__(self):
                    return FakeGetResp()

                async def __aexit__(self, *a):
                    return None

            return _Ctx()

        async def aclose(self):
            pass

    monkeypatch.setattr(stream_service.httpx, "AsyncClient", FakeClient)
    resp = await stream_service._proxy_head_response(
        StreamInfo(url="https://up.example/x.m4a", mime="audio/mp4"),
        range_header="bytes=0-99",
    )
    assert resp.status_code == 206
    assert resp.body == b""
    assert resp.headers["content-range"] == "bytes 0-99/1000"


@pytest.mark.asyncio
async def test_dash_fetch_concat_preserves_order_with_out_of_order_delays(tmp_path):
    delays = {
        "https://cdn/0": 0.05,
        "https://cdn/1": 0.01,
        "https://cdn/2": 0.04,
        "https://cdn/3": 0.00,
        "https://cdn/4": 0.03,
        "https://cdn/5": 0.01,
    }
    payloads = {u: f"[{u.rsplit('/', 1)[-1]}]".encode() for u in delays}

    class FakeSegClient:
        def stream(self, method, url):
            class _R:
                status_code = 200

                def raise_for_status(self):
                    return None

                async def aiter_bytes(self, n):
                    await asyncio.sleep(delays[url])
                    yield payloads[url]

            class _Ctx:
                async def __aenter__(self):
                    return _R()

                async def __aexit__(self, *a):
                    return None

            return _Ctx()

    out = tmp_path / "joined.bin"
    await dash.fetch_concat(FakeSegClient(), list(delays.keys()), out, concurrency=4)
    assert out.read_bytes() == b"[0][1][2][3][4][5]"
