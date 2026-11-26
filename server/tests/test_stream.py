"""Stream service tests: local file 206 Range, remote fallback (needs Postgres)."""

from pathlib import Path

import pytest
from fastapi import HTTPException
from fastapi.responses import Response

from app import db
from app.connectors.base import StreamInfo
from app.services import stream_service


@pytest.fixture(autouse=True)
async def _schema():
    await db.apply_schema()
    yield
    await db.close_engine()


async def _mk_track_with_local(tmpfile: Path) -> int:
    rows = await db.execute_returning(
        "INSERT INTO tracks (title, artists, album, duration_sec) VALUES ('S', '{A}', '', 10) RETURNING id"
    )
    tid = rows[0]["id"]
    await db.execute(
        "INSERT INTO track_sources (track_id, connector, external_id, file_path, mime) "
        "VALUES (:t, 'local', :e, :fp, 'audio/mpeg')",
        {"t": tid, "e": f"local::{tmpfile.name}", "fp": str(tmpfile)},
    )
    return tid


@pytest.mark.asyncio
async def test_local_stream_full(tmp_path):
    f = tmp_path / "a.mp3"
    f.write_bytes(b"ID3" + b"\x00" * 100)
    tid = await _mk_track_with_local(f)
    resp = await stream_service.stream_response(tid, None)
    assert resp.status_code == 200
    assert "audio/mpeg" in resp.media_type
    await db.execute("DELETE FROM tracks WHERE id = :t", {"t": tid})


@pytest.mark.asyncio
async def test_falls_back_to_next_remote_source(monkeypatch):
    """A dead ICM upstream (502) must not end the request: try the next source."""
    rows = await db.execute_returning(
        "INSERT INTO tracks (title, artists, duration_sec) VALUES ('Fallback', '{A}', 10) RETURNING id"
    )
    tid = rows[0]["id"]
    await db.execute(
        """
        INSERT INTO track_sources (track_id, connector, external_id, quality_tier)
        VALUES (:t, 'icm', 'dead', 1), (:t, 'yandex', 'alive', 0)
        """,
        {"t": tid},
    )

    class FakeConnector:
        def needs_assembly(self, info):
            return False

        async def resolve_stream(self, external_id):
            return StreamInfo(url=f"https://example.test/{external_id}", mime="audio/mpeg")

    monkeypatch.setattr(
        stream_service, "get_connector", lambda name: FakeConnector() if name in ("icm", "yandex") else None
    )
    calls: list[str] = []

    async def fake_proxy(info, range_header, *, track_id, connector_name, connector):
        calls.append(connector_name)
        if connector_name == "icm":
            raise HTTPException(502, "upstream stream failed")
        return Response(content=b"ok", media_type="audio/mpeg", status_code=206)

    monkeypatch.setattr(stream_service, "_proxy_response", fake_proxy)

    resp = await stream_service.stream_response(tid, "bytes=0-1")

    assert resp.status_code == 206
    assert calls == ["icm", "yandex"]
    await db.execute("DELETE FROM tracks WHERE id = :t", {"t": tid})


@pytest.mark.asyncio
async def test_corrupt_local_file_falls_through_to_remote(tmp_path, monkeypatch):
    """A measured corrupt local file must not be served with 206 (257 such files)."""
    f = tmp_path / "corrupt.m4a"
    f.write_bytes(b"not really mp4")
    tid = await _mk_track_with_local(f)
    source = await db.fetch_one(
        "SELECT id FROM track_sources WHERE track_id = :t AND connector = 'local'",
        {"t": tid},
    )
    await db.execute(
        """
        INSERT INTO track_quality
            (track_id, local_source_id, file_path, tier, error)
        VALUES (:t, :s, :p, 0, 'truncated')
        """,
        {"t": tid, "s": source["id"], "p": str(f)},
    )
    # no remote connector available in this test -> exhausted after skipping local
    monkeypatch.setattr(stream_service, "get_connector", lambda name: None)
    with pytest.raises(HTTPException) as e:
        await stream_service.stream_response(tid, None)
    assert e.value.status_code == 404
    row = await db.fetch_one(
        "SELECT available FROM track_sources WHERE id = :i", {"i": source["id"]}
    )
    assert row["available"] is True, "quality skip must not mark the source unavailable"
    await db.execute("DELETE FROM tracks WHERE id = :t", {"t": tid})


@pytest.mark.asyncio
async def test_local_stream_missing_file_410_falls_through(tmp_path):
    # local source points to a non-existent file, remote fallback also absent
    f = tmp_path / "ghost.mp3"  # never created
    rows = await db.execute_returning(
        "INSERT INTO tracks (title, artists, album, duration_sec) VALUES ('G', '{A}', '', 10) RETURNING id"
    )
    tid = rows[0]["id"]
    await db.execute(
        "INSERT INTO track_sources (track_id, connector, external_id, file_path) VALUES (:t, 'local', 'g', :fp)",
        {"t": tid, "fp": str(f)},
    )
    # remote source row that will fail to resolve (unknown connector id)
    await db.execute(
        "INSERT INTO track_sources (track_id, connector, external_id) VALUES (:t, 'icm', 'nope999')",
        {"t": tid},
    )
    with pytest.raises(HTTPException) as e:
        await stream_service.stream_response(tid, None)
    # after failed local + failed remote -> 404 (all sources exhausted)
    assert e.value.status_code in (404, 410)
    # local source got marked unavailable
    row = await db.fetch_one("SELECT available FROM track_sources WHERE external_id = 'g'")
    assert row["available"] is False
    await db.execute("DELETE FROM tracks WHERE id = :t", {"t": tid})


@pytest.mark.asyncio
async def test_proxy_relays_range_and_206(monkeypatch):
    """Proxy branch: client Range forwarded upstream, 206 + headers relayed."""
    from app.services import stream_service

    seen: dict = {}

    class FakeUpstream:
        status_code = 206
        headers = {
            "content-range": "bytes 100-1099/50000",
            "content-length": "1000",
            "accept-ranges": "bytes",
            "content-type": "audio/mpeg",
        }

        async def aiter_bytes(self, n):
            yield b"x" * 1000

    class FakeCtx:
        async def __aenter__(self):
            return FakeUpstream()

        async def __aexit__(self, *a):
            return None

    class FakeClient:
        def __init__(self, *a, **kw):
            pass

        def stream(self, method, url, headers=None):
            seen["url"] = url
            seen["headers"] = headers or {}
            return FakeCtx()

        async def aclose(self):
            pass

    monkeypatch.setattr(stream_service.httpx, "AsyncClient", FakeClient)
    info = stream_service.StreamInfo(url="https://up.example/a.mp3", mime="audio/mpeg")
    resp = await stream_service._proxy_response(info, range_header="bytes=100-1099")

    assert seen["headers"]["Range"] == "bytes=100-1099"
    assert resp.status_code == 206
    assert resp.media_type == "audio/mpeg"
    assert resp.headers.get("content-range") == "bytes 100-1099/50000"
    assert resp.headers.get("accept-ranges") == "bytes"
    assert resp.headers.get("content-length") == "1000"


@pytest.mark.asyncio
async def test_proxy_without_range_is_200(monkeypatch):
    from app.services import stream_service

    class FakeUpstream:
        status_code = 200
        headers = {"accept-ranges": "bytes"}

        async def aiter_bytes(self, n):
            yield b"y"

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return None

    class FakeClient:
        def __init__(self, *a, **kw):
            pass

        def stream(self, method, url, headers=None):
            class _Ctx:
                async def __aenter__(self):
                    return FakeUpstream()

                async def __aexit__(self, *a):
                    return None

            return _Ctx()

        async def aclose(self):
            pass

    monkeypatch.setattr(stream_service.httpx, "AsyncClient", FakeClient)
    info = stream_service.StreamInfo(url="https://up.example/b.mp3", mime="audio/mpeg")
    resp = await stream_service._proxy_response(info, range_header=None)
    assert resp.status_code == 200


class _Resp:
    def __init__(self, status, headers, body):
        self.status_code = status
        self.headers = headers
        self._body = body

    async def aiter_bytes(self, n):
        yield self._body


def _fake_client(resp):
    class FakeClient:
        def __init__(self, *a, **kw):
            pass

        def stream(self, method, url, headers=None):
            class _Ctx:
                async def __aenter__(self):
                    return resp

                async def __aexit__(self, *a):
                    return None

            return _Ctx()

        async def aclose(self):
            pass

    return FakeClient


async def _consume(resp):
    async for _ in resp.body_iterator:
        pass


@pytest.mark.asyncio
async def test_partial_range_is_not_cached_as_full(monkeypatch):
    """Range: bytes=0-1023 (206 partial) must NOT poison the cache as a full file."""
    from app.services import stream_service

    monkeypatch.setattr(
        stream_service.httpx,
        "AsyncClient",
        _fake_client(
            _Resp(
                206,
                {
                    "content-range": "bytes 0-1023/5000000",
                    "content-length": "1024",
                    "accept-ranges": "bytes",
                    "content-type": "audio/mpeg",
                },
                b"x" * 1024,
            )
        ),
    )
    tid = 991234
    info = stream_service.StreamInfo(url="https://up.example/p.mp3", mime="audio/mpeg")
    resp = await stream_service._proxy_response(info, range_header="bytes=0-1023", track_id=tid)
    await _consume(resp)

    cdir = stream_service.stream_cache_dir()
    assert not (cdir / f"{tid}.mp3").exists(), "частичный Range не должен попадать в кэш"
    assert not list(cdir.glob(f"{tid}_*.part"))


@pytest.mark.asyncio
async def test_full_200_is_cached(monkeypatch):
    """A complete 200 body is cached as the full file."""
    from app.services import stream_service

    monkeypatch.setattr(
        stream_service.httpx,
        "AsyncClient",
        _fake_client(
            _Resp(200, {"content-length": "10", "content-type": "audio/mpeg"}, b"y" * 10)
        ),
    )
    tid = 992001
    info = stream_service.StreamInfo(url="https://up.example/f.mp3", mime="audio/mpeg")
    resp = await stream_service._proxy_response(info, range_header=None, track_id=tid)
    await _consume(resp)

    cdir = stream_service.stream_cache_dir()
    final = cdir / f"{tid}.mp3"
    assert final.exists(), "полный 200 должен кэшироваться"
    assert final.stat().st_size == 10
    final.unlink(missing_ok=True)


@pytest.mark.asyncio
async def test_full_206_is_cached(monkeypatch):
    """A 206 that covers the whole file (end+1 == total) is cached as complete."""
    from app.services import stream_service

    monkeypatch.setattr(
        stream_service.httpx,
        "AsyncClient",
        _fake_client(
            _Resp(
                206,
                {
                    "content-range": "bytes 0-9/10",
                    "content-length": "10",
                    "content-type": "audio/mpeg",
                },
                b"z" * 10,
            )
        ),
    )
    tid = 992002
    info = stream_service.StreamInfo(url="https://up.example/g.mp3", mime="audio/mpeg")
    resp = await stream_service._proxy_response(info, range_header="bytes=0-", track_id=tid)
    await _consume(resp)

    cdir = stream_service.stream_cache_dir()
    final = cdir / f"{tid}.mp3"
    assert final.exists(), "полный 206 должен кэшироваться"
    assert final.stat().st_size == 10
    final.unlink(missing_ok=True)


def test_safe_url_strips_signed_query():
    from app.services.stream_service import _safe_url

    assert _safe_url("https://up.example/a.mp3?token=SECRET&sig=ABC") == "https://up.example/a.mp3"
    assert _safe_url("https://up.example/a.mp3") == "https://up.example/a.mp3"
    assert _safe_url("https://up.example/a.mp3#frag?x=1") == "https://up.example/a.mp3"


class _ResolveConnector:
    def __init__(self, info: StreamInfo):
        self.info = info
        self.calls = 0

    async def resolve_stream(self, external_id: str):
        self.calls += 1
        return self.info


@pytest.mark.asyncio
async def test_resolve_cache_reuses_direct_url_until_forgotten():
    stream_service._resolve_cache.clear()
    conn = _ResolveConnector(StreamInfo(url="https://up.example/a.mp3"))

    first = await stream_service._resolve_cached("yandex", conn, "1")
    second = await stream_service._resolve_cached("yandex", conn, "1")
    assert first is second
    assert conn.calls == 1

    stream_service._forget_resolved("yandex", "1")
    await stream_service._resolve_cached("yandex", conn, "1")
    assert conn.calls == 2


@pytest.mark.asyncio
async def test_resolve_cache_skips_assembled_streams():
    stream_service._resolve_cache.clear()
    conn = _ResolveConnector(
        StreamInfo(url="https://up.example/m.mpd", transform="dash_concat")
    )

    await stream_service._resolve_cached("tidal", conn, "2")
    await stream_service._resolve_cached("tidal", conn, "2")
    assert conn.calls == 2
