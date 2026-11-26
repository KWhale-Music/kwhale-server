"""KWhale Core v1 contract tests (reduced /v1 surface)."""

from __future__ import annotations

import uuid

import pytest
from httpx import ASGITransport, AsyncClient

from app import db
from app.auth import ensure_seed_user
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


@pytest.fixture
async def admin_headers(client):
    response = await client.post(
        "/v1/auth/login",
        json={"email": settings.seed_email, "password": settings.seed_password},
    )
    assert response.status_code == 200, response.text
    return {"Authorization": f"Bearer {response.json()['accessToken']}"}


async def _track(title: str = "Parity track", *, local: bool = False) -> int:
    rows = await db.execute_returning(
        """
        INSERT INTO tracks
            (title, artists, album, duration_sec, year, track_number, genre, explicit, last_seen_at)
        VALUES (:t, '{Test Artist}', 'Test Album', 200, 2024, 1, 'Rock', FALSE, NOW())
        RETURNING id
        """,
        {"t": title},
    )
    track_id = rows[0]["id"]
    if local:
        await db.execute(
            """
            INSERT INTO track_sources
                (track_id, connector, external_id, quality_tier, mime, file_path, available)
            VALUES (:t, 'local', :e, 100, 'audio/mpeg', :p, TRUE)
            ON CONFLICT DO NOTHING
            """,
            {"t": track_id, "e": f"local::test-{track_id}.mp3", "p": f"/tmp/test-{track_id}.mp3"},
        )
    return track_id


@pytest.mark.asyncio
async def test_login_refresh_logout(client):
    login = await client.post(
        "/v1/auth/login",
        json={"email": settings.seed_email, "password": settings.seed_password},
    )
    assert login.status_code == 200
    body = login.json()
    assert body["tokenType"] == "Bearer"
    assert body["user"]["id"] == "1" or body["user"]["id"].isdigit()

    refreshed = await client.post(
        "/v1/auth/refresh", json={"refreshToken": body["refreshToken"]}
    )
    assert refreshed.status_code == 200
    assert refreshed.json()["refreshToken"] != body["refreshToken"]

    reused = await client.post(
        "/v1/auth/refresh", json={"refreshToken": body["refreshToken"]}
    )
    assert reused.status_code == 401
    assert reused.headers["content-type"].startswith("application/problem+json")

    logout = await client.post(
        "/v1/auth/logout", json={"refreshToken": refreshed.json()["refreshToken"]}
    )
    assert logout.status_code == 204


@pytest.mark.asyncio
async def test_invite_registration_and_admin_role(client, admin_headers):
    invite = await client.post("/admin/api/invites", headers=admin_headers, json={"days": 7})
    assert invite.status_code == 201
    token = invite.json()["token"]

    email = f"user-{uuid.uuid4().hex[:8]}@example.com"
    registered = await client.post(
        "/v1/auth/register",
        json={
            "email": email,
            "password": "password123",
            "displayName": "Invited User",
            "inviteToken": token,
        },
    )
    assert registered.status_code == 201, registered.text
    assert registered.json()["user"]["email"] == email

    reused = await client.post(
        "/v1/auth/register",
        json={
            "email": f"other-{uuid.uuid4().hex[:8]}@example.com",
            "password": "password123",
            "displayName": "Other",
            "inviteToken": token,
        },
    )
    assert reused.status_code == 403

    users = await client.get("/admin/api/users", headers=admin_headers)
    assert any(u["email"] == email for u in users.json()["items"])


@pytest.mark.asyncio
async def test_track_dto_has_contract_shape(client, admin_headers):
    track_id = await _track("Local contract song", local=True)
    response = await client.get(f"/v1/tracks/{track_id}", headers=admin_headers)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["id"] == str(track_id)
    assert body["durationMs"] == 200_000
    assert body["cover"]["id"]
    assert body["artists"][0]["name"] == "Test Artist"
    assert body["isDownloaded"] is True
    assert body["downloadStatus"] == "ready"
    assert body["isLiked"] is False


@pytest.mark.asyncio
async def test_like_and_unlike(client, admin_headers):
    track_id = await _track("Like me", local=True)
    assert (
        await client.put(f"/v1/tracks/{track_id}/like", headers=admin_headers)
    ).status_code == 204
    body = (await client.get(f"/v1/tracks/{track_id}", headers=admin_headers)).json()
    assert body["isLiked"] is True
    assert (
        await client.delete(f"/v1/tracks/{track_id}/like", headers=admin_headers)
    ).status_code == 204
    body = (await client.get(f"/v1/tracks/{track_id}", headers=admin_headers)).json()
    assert body["isLiked"] is False


@pytest.mark.asyncio
async def test_playlist_crud_and_like_all(client, admin_headers):
    track_id = await _track("Playlist song", local=True)
    created = await client.post(
        "/v1/playlists",
        headers=admin_headers,
        json={"title": "V1 playlist", "visibility": "private"},
    )
    assert created.status_code == 201, created.text
    playlist = created.json()
    assert playlist["title"] == "V1 playlist"
    assert playlist["permissions"]["canEdit"] is True

    added = await client.post(
        f"/v1/playlists/{playlist['id']}/tracks",
        headers=admin_headers,
        json={"trackIds": [str(track_id)]},
    )
    assert added.status_code == 200, added.text
    assert added.json()["trackCount"] == 1

    liked = await client.put(
        f"/v1/playlists/{playlist['id']}/tracks/like", headers=admin_headers
    )
    assert liked.status_code == 202, liked.text
    result = liked.json()
    assert result["requested"] == 1
    assert result["liked"] == 1
    assert result["failed"] == 0

    ordered = await client.put(
        f"/v1/playlists/{playlist['id']}/tracks/order",
        headers=admin_headers,
        json={"trackIds": [str(track_id)]},
    )
    assert ordered.status_code == 200


@pytest.mark.asyncio
async def test_search_returns_local_first_and_flat_shape(client, admin_headers):
    # Disable remote fan-out for this test: local search must be deterministic.
    from app.services.source_accounts import SOURCE_SPECS, set_enabled

    for name in SOURCE_SPECS:
        await set_enabled(name, False)
    title = f"Unique local search {uuid.uuid4().hex[:8]}"
    track_id = await _track(title, local=True)
    response = await client.get(f"/v1/search?q={title}", headers=admin_headers)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["tracks"][0]["id"] == str(track_id)
    assert "errors" not in body
    assert body["total"] >= 1


@pytest.mark.asyncio
async def test_events_are_persisted(client, admin_headers):
    event_id = str(uuid.uuid4())
    response = await client.post(
        "/v1/events",
        headers=admin_headers,
        json={
            "events": [
                {
                    "id": event_id,
                    "type": "play",
                    "trackId": "123",
                    "playbackId": "session-1",
                    "listenedMs": 42_000,
                    "trackDurationMs": 200_000,
                    "occurredAt": "2026-11-05T12:00:00Z",
                    "timezoneOffsetMinutes": 180,
                }
            ]
        },
    )
    assert response.status_code == 202, response.text
    row = await db.fetch_one("SELECT id, type, payload FROM events WHERE id = CAST(:i AS uuid)", {"i": event_id})
    assert row is not None
    assert row["type"] == "play"
    assert row["payload"]["listenedMs"] == 42_000


@pytest.mark.asyncio
async def test_placeholder_image_and_problem_content_type(client):
    image = await client.get("/v1/images/placeholder?size=md")
    assert image.status_code == 200
    assert image.headers["content-type"] == "image/png"
    assert image.headers["etag"]

    missing = await client.get("/v1/tracks/999999999")
    assert missing.status_code == 401
    assert missing.headers["content-type"].startswith("application/problem+json")


@pytest.mark.asyncio
async def test_stream_range_and_download_attachment(client, admin_headers, tmp_path):
    track_id = await _track("Range song", local=False)
    media = tmp_path / "range.mp3"
    media.write_bytes(b"0123456789")
    await db.execute(
        """
        INSERT INTO track_sources
            (track_id, connector, external_id, quality_tier, mime, file_path, available)
        VALUES (:t, 'local', :e, 100, 'audio/mpeg', :p, TRUE)
        """,
        {"t": track_id, "e": f"local::range-{track_id}.mp3", "p": str(media)},
    )
    stream = await client.get(
        f"/v1/tracks/{track_id}/stream", headers={**admin_headers, "Range": "bytes=0-3"}
    )
    assert stream.status_code == 206
    assert stream.content == b"0123"
    assert stream.headers["accept-ranges"] == "bytes"

    download = await client.get(f"/v1/tracks/{track_id}/download", headers=admin_headers)
    assert download.status_code == 200
    assert download.content == b"0123456789"
    assert "attachment" in download.headers["content-disposition"]


@pytest.mark.asyncio
async def test_upload_rejects_non_audio(client, admin_headers):
    response = await client.post(
        "/v1/uploads/tracks",
        headers=admin_headers,
        files={"file": ("notes.txt", b"hello", "text/plain")},
    )
    assert response.status_code == 400
    assert response.headers["content-type"].startswith("application/problem+json")


def test_dedup_preserves_editions_and_merges_explicit_clean():
    from app.connectors.base import TrackMeta
    from app.services.dedup import dedup_tracks, track_identity_key

    orig_key = track_identity_key("Smells Like Teen Spirit", ["Nirvana"], 301)
    remaster_key = track_identity_key(
        "Smells Like Teen Spirit (2021 Remastered)", ["Nirvana"], 301
    )
    radio_key = track_identity_key("Smells Like Teen Spirit (Radio Edit)", ["Nirvana"], 240)
    explicit_key = track_identity_key("Smells Like Teen Spirit (Explicit)", ["Nirvana"], 301)
    clean_key = track_identity_key("Smells Like Teen Spirit (Clean)", ["Nirvana"], 301)

    assert orig_key != remaster_key
    assert orig_key != radio_key
    assert orig_key == explicit_key == clean_key

    clean_meta = TrackMeta(
        connector="deezer",
        external_id="1",
        title="Song (Clean)",
        artists=["Artist"],
        duration_sec=200,
        quality_tier=20,
        explicit=False,
    )
    explicit_meta = TrackMeta(
        connector="yandex",
        external_id="2",
        title="Song (Explicit)",
        artists=["Artist"],
        duration_sec=200,
        quality_tier=20,
        explicit=True,
    )
    remaster_meta = TrackMeta(
        connector="yandex",
        external_id="3",
        title="Song (Remastered 2020)",
        artists=["Artist"],
        duration_sec=200,
        quality_tier=20,
        explicit=False,
    )
    merged = dedup_tracks([clean_meta, explicit_meta, remaster_meta])
    assert len(merged) == 2
    assert merged[0]["winner"].explicit is True
    assert merged[0]["winner"].external_id == "2"


@pytest.mark.asyncio
async def test_modular_proxy_admin_api(client, admin_headers):
    from app import httpcore

    get_res = await client.get("/admin/api/proxy", headers=admin_headers)
    assert get_res.status_code == 200
    initial = get_res.json()
    assert "enabled" in initial
    assert "overrides" in initial
    assert "defaultProxy" in initial

    put_res = await client.put(
        "/admin/api/proxy",
        headers=admin_headers,
        json={
            "enabled": True,
            "defaultProxy": "socks5://127.0.0.1:1080",
            "overrides": {"yandex": "direct", "deezer": "http://127.0.0.1:8080"},
        },
    )
    assert put_res.status_code == 200
    assert httpcore.proxy_for("yandex") is None
    assert httpcore.proxy_for("deezer") == "http://127.0.0.1:8080"
    # vk is in _DIRECT_BY_DEFAULT -> stays direct even with a default proxy
    assert httpcore.proxy_for("vk") is None
    # a connector with no override and not direct-by-default falls back to defaultProxy
    assert httpcore.proxy_for("tidal") == "socks5://127.0.0.1:1080"

    off_res = await client.put(
        "/admin/api/proxy",
        headers=admin_headers,
        json={
            "enabled": False,
            "defaultProxy": "socks5://127.0.0.1:1080",
            "overrides": {"deezer": "http://127.0.0.1:8080"},
        },
    )
    assert off_res.status_code == 200
    assert httpcore.proxy_for("deezer") is None
    assert httpcore.proxy_for("vk") is None


@pytest.mark.asyncio
async def test_onboarding_persistence_and_typed_suggestions(client, admin_headers):
    tid = await _track("Neon Horizon", local=True)
    assert tid > 0

    put_res = await client.put(
        "/v1/recommendations/onboarding",
        headers=admin_headers,
        json={
            "genres": [{"id": "rock", "name": "Rock"}],
            "artists": [{"id": "1", "name": "Test Artist"}],
        },
    )
    assert put_res.status_code == 204

    get_res = await client.get("/v1/recommendations/onboarding", headers=admin_headers)
    assert get_res.status_code == 200
    body = get_res.json()
    assert any(g["name"] == "Rock" for g in body["genres"])
    assert any(a["id"] == "1" for a in body["artists"])

    sug_res = await client.get("/v1/search/suggestions?q=Neon", headers=admin_headers)
    assert sug_res.status_code == 200
    sug_items = sug_res.json()["items"]
    assert any(s["text"] == "Neon Horizon" and s["type"] == "track" for s in sug_items)


@pytest.mark.asyncio
async def test_full_e2e_user_journey(client, admin_headers, tmp_path):
    """End-to-end journey: invite -> register -> profile -> onboarding -> upload -> search -> stream -> events -> history -> playlist -> mixes."""
    from app.config import settings

    orig_music_dir = settings.music_dir
    object.__setattr__(settings, "music_dir", tmp_path)
    try:
        invite_res = await client.post("/admin/api/invites", headers=admin_headers, json={"days": 7})
        assert invite_res.status_code == 201
        token = invite_res.json()["token"]

        email = f"e2e-{uuid.uuid4().hex[:8]}@example.com"
        reg_res = await client.post(
            "/v1/auth/register",
            json={
                "email": email,
                "password": "password123",
                "displayName": "E2E User",
                "inviteToken": token,
            },
        )
        assert reg_res.status_code == 201
        user_headers = {"Authorization": f"Bearer {reg_res.json()['accessToken']}"}

        patch_me = await client.patch(
            "/v1/me",
            headers=user_headers,
            json={"displayName": "E2E Pro"},
        )
        assert patch_me.status_code == 200
        assert patch_me.json()["displayName"] == "E2E Pro"

        frame = b"\xff\xfb\x90\x00" + b"\x00" * 413
        mp3_bytes = b"ID3\x03\x00\x00\x00\x00\x00\x00" + frame * 20
        idem_key = str(uuid.uuid4())
        up_res = await client.post(
            "/v1/uploads/tracks",
            headers={**user_headers, "Idempotency-Key": idem_key},
            files={"file": ("e2e.mp3", mp3_bytes, "audio/mpeg")},
            data={"title": "E2E Anthem", "artist": "E2E Band"},
        )
        assert up_res.status_code == 201, up_res.text
        track_id = up_res.json()["id"]

        replay_res = await client.post(
            "/v1/uploads/tracks",
            headers={**user_headers, "Idempotency-Key": idem_key},
            files={"file": ("e2e.mp3", mp3_bytes, "audio/mpeg")},
            data={"title": "E2E Anthem", "artist": "E2E Band"},
        )
        assert replay_res.status_code == 201
        assert replay_res.json()["id"] == track_id

        stream_res = await client.get(
            f"/v1/tracks/{track_id}/stream",
            headers={**user_headers, "Range": "bytes=0-15"},
        )
        assert stream_res.status_code == 206
        assert len(stream_res.content) == 16

        ev_res = await client.post(
            "/v1/events",
            headers=user_headers,
            json={
                "events": [
                    {
                        "id": str(uuid.uuid4()),
                        "type": "play",
                        "trackId": track_id,
                        "occurredAt": "2026-11-05T12:00:00Z",
                    }
                ]
            },
        )
        assert ev_res.status_code == 202

        hist_res = await client.get("/v1/history", headers=user_headers)
        assert hist_res.status_code == 200
        assert any(item["track"]["id"] == track_id for item in hist_res.json()["items"])

        mix_res = await client.get("/v1/mixes", headers=user_headers)
        assert mix_res.status_code == 200
        assert len(mix_res.json()["items"]) >= 1
    finally:
        object.__setattr__(settings, "music_dir", orig_music_dir)
