"""Admin-issued external embed widgets: Yandex-style iframe cards, Edge+Home warm, CSP, GC."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from httpx import ASGITransport, AsyncClient

from app import auth, db
from app.config import settings
from app.main import app
from app.services import embed_service, share_card
from app.services import gc as gc_service


@pytest.fixture(autouse=True)
async def _schema(tmp_path):
    auth.clear_user_cache()
    share_card._card_cache.clear()
    share_card._palette_cache.clear()
    prev_music = settings.music_dir
    object.__setattr__(settings, "music_dir", tmp_path / "music")
    settings.music_dir.mkdir(parents=True, exist_ok=True)
    await db.apply_schema()
    await auth.ensure_seed_user()
    yield
    await embed_service.aclose_background_tasks()
    auth.clear_user_cache()
    object.__setattr__(settings, "music_dir", prev_music)
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


async def _user_headers(client) -> dict[str, str]:
    admin_login = await client.post(
        "/v1/auth/login",
        json={"email": settings.seed_email, "password": settings.seed_password},
    )
    admin_hdr = {"Authorization": f"Bearer {admin_login.json()['accessToken']}"}
    inv_r = await client.post("/admin/api/invites", headers=admin_hdr, json={"days": 1})
    token = inv_r.json()["token"]
    reg = await client.post(
        "/v1/auth/register",
        json={
            "inviteToken": token,
            "email": "plain-user@example.com",
            "password": "user-pass-1234",
            "displayName": "Plain User",
        },
    )
    assert reg.status_code == 201
    return {"Authorization": f"Bearer {reg.json()['accessToken']}"}


async def _seed_catalog(tmp_path: Path) -> dict[str, Any]:
    await db.execute("DELETE FROM embed_widget_tracks")
    await db.execute("DELETE FROM embed_widgets")
    await db.execute("DELETE FROM download_jobs")
    await db.execute("DELETE FROM share_clicks")
    await db.execute("DELETE FROM share_links")
    await db.execute("DELETE FROM track_sources")
    await db.execute("DELETE FROM album_sources")
    await db.execute("DELETE FROM artist_sources")
    await db.execute("DELETE FROM tracks")
    await db.execute("DELETE FROM albums")
    await db.execute("DELETE FROM artists")

    art = await db.execute_returning(
        """
        INSERT INTO artists (name, cover_url, genre, last_seen_at)
        VALUES ('Kino Embed', 'https://example.org/kino.jpg', 'Рок', NOW())
        RETURNING id
        """
    )
    artist_id = int(art[0]["id"])

    alb = await db.execute_returning(
        """
        INSERT INTO albums (title, artist_id, year, track_count, cover_url, last_seen_at)
        VALUES ('Группа крови', :a, 1988, 2, 'https://example.org/album.jpg', NOW())
        RETURNING id
        """,
        {"a": artist_id},
    )
    album_id = int(alb[0]["id"])

    t1 = await db.execute_returning(
        """
        INSERT INTO tracks (title, artists, album, duration_sec, track_number, cover_url, last_seen_at)
        VALUES ('Группа крови', ARRAY['Kino Embed'], 'Группа крови', 284, 1, 'https://example.org/album.jpg', NOW())
        RETURNING id
        """
    )
    t2 = await db.execute_returning(
        """
        INSERT INTO tracks (title, artists, album, duration_sec, track_number, cover_url, last_seen_at)
        VALUES ('Звезда по имени Солнце', ARRAY['Kino Embed'], 'Группа крови', 225, 2, 'https://example.org/album.jpg', NOW())
        RETURNING id
        """
    )
    t1_id = int(t1[0]["id"])
    t2_id = int(t2[0]["id"])

    local_file = tmp_path / "music" / "kino_1.mp3"
    local_file.parent.mkdir(parents=True, exist_ok=True)
    local_file.write_bytes(b"ID3" + b"\x00" * 1024)

    await db.execute(
        """
        INSERT INTO track_sources (track_id, connector, external_id, quality_tier, available, file_path)
        VALUES (:t1, 'local', 'local::kino1', 100, TRUE, :fp),
               (:t2, 'yandex', '987654', 65, TRUE, NULL)
        """,
        {"t1": t1_id, "t2": t2_id, "fp": str(local_file)},
    )
    return {
        "artist_id": artist_id,
        "album_id": album_id,
        "t1_id": t1_id,
        "t2_id": t2_id,
        "local_file": str(local_file),
    }


@pytest.mark.asyncio
async def test_embed_admin_guard_search_and_yandex_card_rendering(client, admin_headers, tmp_path):
    ids = await _seed_catalog(tmp_path)
    user_hdr = await _user_headers(client)

    # Non-admin users cannot list, search, or issue embeds.
    assert (await client.get("/admin/api/embeds", headers=user_hdr)).status_code == 403
    assert (
        await client.post(
            "/admin/api/embeds",
            headers=user_hdr,
            json={"kind": "track", "entityId": ids["t1_id"]},
        )
    ).status_code == 403

    # Search finds tracks/albums/artists by text and exact ID.
    sr_track = await client.get(
        f"/admin/api/embeds/search?kind=track&q={ids['t2_id']}", headers=admin_headers
    )
    assert sr_track.status_code == 200
    assert sr_track.json()["items"][0]["entityId"] == str(ids["t2_id"])
    assert sr_track.json()["items"][0]["localReady"] is False

    sr_album = await client.get(
        "/admin/api/embeds/search?kind=album&q=Группа", headers=admin_headers
    )
    assert sr_album.status_code == 200
    assert sr_album.json()["items"][0]["entityId"] == str(ids["album_id"])

    for k in ("track", "album", "artist"):
        sr_empty = await client.get(f"/admin/api/embeds/search?kind={k}&q=", headers=admin_headers)
        assert sr_empty.status_code == 200
        assert len(sr_empty.json()["items"]) >= 1

    sr_artist = await client.get(
        "/admin/api/embeds/search?kind=artist&q=Kino", headers=admin_headers
    )
    assert sr_artist.status_code == 200
    assert sr_artist.json()["items"][0]["entityId"] == str(ids["artist_id"])

    # Issue album embed with allowed domains (no network edge warm URL set).
    created = await client.post(
        "/admin/api/embeds",
        headers=admin_headers,
        json={
            "kind": "album",
            "entityId": ids["album_id"],
            "theme": "dark",
            "allowedOrigins": ["example.com", "*.blog.org"],
        },
    )
    assert created.status_code == 201
    widget = created.json()["item"]
    wid = widget["id"]
    assert widget["kind"] == "album"
    assert widget["cache"]["total"] == 2
    assert widget["cache"]["homeReady"] == 1
    assert widget["cache"]["homePending"] == 1  # t2 queued to download_jobs for Home

    # Pending download job was created for non-local track t2.
    dj = await db.fetch_one(
        "SELECT status, target_type FROM download_jobs WHERE track_id = :t",
        {"t": ids["t2_id"]},
    )
    assert dj is not None
    assert dj["status"] == "pending"
    assert dj["target_type"] == "embed"

    # Public embed page renders Yandex-style tracklist card with CSP & no-store.
    page = await client.get(f"/share/embed/{wid}")
    assert page.status_code == 200
    assert page.headers["cache-control"] == "no-store"
    assert "x-frame-options" not in {k.lower() for k in page.headers}
    csp = page.headers["content-security-policy"]
    assert "frame-ancestors 'self' example.com *.blog.org" == csp
    assert "Группа крови" in page.text
    assert "Звезда по имени Солнце" in page.text
    assert f"/v1/tracks/{ids['t1_id']}/stream?exp=" in page.text
    assert f"/v1/tracks/{ids['t2_id']}/stream?exp=" in page.text

    # Stream URL refresh works ONLY for tracks belonging to this widget.
    ok_stream = await client.get(f"/share/embed/{wid}/stream/{ids['t1_id']}")
    assert ok_stream.status_code == 200
    signed_url = ok_stream.json()["streamUrl"]
    assert signed_url.startswith(f"/v1/tracks/{ids['t1_id']}/stream?exp=")

    # Signed stream URL plays audio Range 206 directly without Bearer JWT.
    stream_resp = await client.get(signed_url, headers={"Range": "bytes=0-15"})
    assert stream_resp.status_code == 206
    assert len(stream_resp.content) == 16

    # Arbitrary track ID outside the widget is denied (404).
    other = await db.execute_returning(
        "INSERT INTO tracks (title, artists, last_seen_at) VALUES ('Other', ARRAY['X'], NOW()) RETURNING id"
    )
    assert (await client.get(f"/share/embed/{wid}/stream/{other[0]['id']}")).status_code == 404

    # Clicking "Open in KWhale" lazily creates/refreshes /share/s/{id}.
    open_r = await client.get(f"/share/embed/{wid}/open", follow_redirects=False)
    assert open_r.status_code == 302
    assert "/share/s/" in open_r.headers["location"]

    # Artist embed renders circular artist card + tracks.
    art_res = await client.post(
        "/admin/api/embeds",
        headers=admin_headers,
        json={"kind": "artist", "entityId": ids["artist_id"]},
    )
    assert art_res.status_code == 201
    art_page = await client.get(f"/share/embed/{art_res.json()['item']['id']}")
    assert art_page.status_code == 200
    assert "is-artist" in art_page.text


@pytest.mark.asyncio
async def test_embed_origins_validation_revoke_and_delete(client, admin_headers, tmp_path):
    ids = await _seed_catalog(tmp_path)

    bad = await client.post(
        "/admin/api/embeds",
        headers=admin_headers,
        json={
            "kind": "track",
            "entityId": ids["t1_id"],
            "allowedOrigins": "evil.com; script-src 'unsafe-inline'",
        },
    )
    assert bad.status_code == 422

    res = await client.post(
        "/admin/api/embeds",
        headers=admin_headers,
        json={"kind": "track", "entityId": ids["t1_id"], "allowedOrigins": ""},
    )
    assert res.status_code == 201
    wid = res.json()["item"]["id"]

    # Empty origins -> wildcard allowed.
    p1 = await client.get(f"/share/embed/{wid}")
    assert p1.status_code == 200
    assert p1.headers["content-security-policy"] == "frame-ancestors *"

    # Restrict domains & disable -> immediately 410 on page & stream refresh.
    upd = await client.patch(
        f"/admin/api/embeds/{wid}",
        headers=admin_headers,
        json={"allowedOrigins": "https://music-blog.ru", "active": False},
    )
    assert upd.status_code == 200
    assert upd.json()["item"]["active"] is False
    assert (await client.get(f"/share/embed/{wid}")).status_code == 410
    assert (await client.get(f"/share/embed/{wid}/stream/{ids['t1_id']}")).status_code == 410

    # Re-enable -> 200 with updated frame-ancestors.
    upd2 = await client.patch(
        f"/admin/api/embeds/{wid}",
        headers=admin_headers,
        json={"active": True},
    )
    assert upd2.status_code == 200
    p2 = await client.get(f"/share/embed/{wid}")
    assert p2.status_code == 200
    assert p2.headers["content-security-policy"] == "frame-ancestors 'self' https://music-blog.ru"

    # Delete -> 204 and 404 afterwards.
    assert (await client.delete(f"/admin/api/embeds/{wid}", headers=admin_headers)).status_code == 204
    assert (await client.get(f"/share/embed/{wid}")).status_code == 404


@pytest.mark.asyncio
async def test_embed_edge_warm_and_post_download_hook(client, admin_headers, tmp_path):
    ids = await _seed_catalog(tmp_path)
    seen_urls: list[str] = []

    class _FakeResp:
        def __init__(self, status_code: int = 200):
            self.status_code = status_code

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def aiter_bytes(self, _chunk_size: int = 65536):
            yield b"abc"

    class _FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        def stream(self, _method: str, url: str):
            seen_urls.append(url)
            return _FakeResp(200)

    created = await client.post(
        "/admin/api/embeds",
        headers=admin_headers,
        json={
            "kind": "album",
            "entityId": ids["album_id"],
            "edgeBaseUrl": "https://edge.example.test",
        },
    )
    assert created.status_code == 201
    wid = created.json()["item"]["id"]

    with patch.object(embed_service.httpx, "AsyncClient", _FakeAsyncClient):
        stats = await embed_service.warm_widget_edge(wid, force=True)
    assert stats["warmed"] == 1  # only t1 is local so far
    assert any(f"https://edge.example.test/v1/tracks/{ids['t1_id']}/stream?exp=" in u for u in seen_urls)
    assert any("https://edge.example.test/v1/images/" in u for u in seen_urls)

    # Now simulate t2 completing download on Home -> downloader hook warms t2 on Edge.
    seen_urls.clear()
    await db.execute(
        "UPDATE track_sources SET connector = 'local', file_path = :fp WHERE track_id = :t",
        {"t": ids["t2_id"], "fp": ids["local_file"]},
    )
    with patch.object(embed_service.httpx, "AsyncClient", _FakeAsyncClient):
        await embed_service.warm_track_for_active_widgets(ids["t2_id"])
    assert any(f"https://edge.example.test/v1/tracks/{ids['t2_id']}/stream?exp=" in u for u in seen_urls)

    refreshed = await client.get(f"/admin/api/embeds/{wid}", headers=admin_headers)
    assert refreshed.json()["item"]["cache"]["homeReady"] == 2
    assert refreshed.json()["item"]["cache"]["edgeWarmed"] == 2


@pytest.mark.asyncio
async def test_embed_pins_tracks_files_entities_and_covers_against_gc(client, admin_headers, tmp_path):
    ids = await _seed_catalog(tmp_path)
    res = await client.post(
        "/admin/api/embeds",
        headers=admin_headers,
        json={"kind": "album", "entityId": ids["album_id"]},
    )
    wid = res.json()["item"]["id"]

    # Backdate tracks/albums/artists/images and mark local track with expired pending_delete_since.
    await db.execute(
        "UPDATE tracks SET last_seen_at = NOW() - INTERVAL '60 days' WHERE id IN (:t1, :t2)",
        {"t1": ids["t1_id"], "t2": ids["t2_id"]},
    )
    await db.execute(
        "UPDATE track_sources SET pending_delete_since = NOW() - INTERVAL '60 days' WHERE track_id = :t1 AND connector = 'local'",
        {"t1": ids["t1_id"]},
    )
    await db.execute(
        "UPDATE albums SET last_seen_at = NOW() - INTERVAL '60 days' WHERE id = :a",
        {"a": ids["album_id"]},
    )
    await db.execute(
        "UPDATE artists SET last_seen_at = NOW() - INTERVAL '60 days' WHERE id = :a",
        {"a": ids["artist_id"]},
    )
    await db.execute(
        "UPDATE images SET created_at = NOW() - INTERVAL '60 days', last_accessed_at = NOW() - INTERVAL '60 days'"
    )

    stats = await gc_service.run_gc()
    assert stats["tracks_deleted"] == 0
    assert stats["files_deleted"] == 0
    assert Path(ids["local_file"]).is_file()
    assert (await db.fetch_one("SELECT 1 FROM albums WHERE id = :a", {"a": ids["album_id"]})) is not None

    # After widget is deleted, GC cleans up expired unliked ephemeral tracks/files/albums normally.
    await client.delete(f"/admin/api/embeds/{wid}", headers=admin_headers)
    stats_after = await gc_service.run_gc()
    assert stats_after["files_deleted"] >= 1
    assert stats_after["tracks_deleted"] >= 1
    assert not Path(ids["local_file"]).is_file()
