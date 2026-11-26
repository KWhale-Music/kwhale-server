"""Library entity actions: album/artist cards and followed playlists.

These routes are deliberately separate from "like all tracks":
  * PUT/DELETE /albums/{id}/like  — save the card only (no downloads)
  * PUT/DELETE /artists/{id}/like — save the card only (no downloads)
  * GET /library/albums           — only explicitly saved album cards
  * GET /library/playlists        — own + followed playlists
"""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from app import db
from app.auth import ensure_seed_user
from app.config import settings
from app.main import app
from app.services import playlists as playlists_service
from app.services.entity_upsert import ensure_entities_for_track


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
async def auth_headers(client):
    response = await client.post(
        "/v1/auth/login",
        json={"email": settings.seed_email, "password": settings.seed_password},
    )
    assert response.status_code == 200, response.text
    return {"Authorization": f"Bearer {response.json()['accessToken']}"}


async def _entities(prefix: str) -> tuple[int, int, int]:
    rows = await db.execute_returning(
        """
        INSERT INTO tracks (title, artists, album, duration_sec)
        VALUES (:t, ARRAY[:a], :al, 180)
        RETURNING id
        """,
        {"t": f"{prefix} Track", "a": f"{prefix} Artist", "al": f"{prefix} Album"},
    )
    track_id = rows[0]["id"]
    await ensure_entities_for_track(track_id)
    artist = await db.fetch_one("SELECT id FROM artists WHERE name = :n", {"n": f"{prefix} Artist"})
    album = await db.fetch_one(
        """
        SELECT a.id FROM albums a JOIN artists ar ON ar.id = a.artist_id
        WHERE a.title = :t AND ar.name = :n LIMIT 1
        """,
        {"t": f"{prefix} Album", "n": f"{prefix} Artist"},
    )
    assert artist is not None and album is not None
    return track_id, int(artist["id"]), int(album["id"])


async def _cleanup_entities(track_id: int, artist_id: int, album_id: int) -> None:
    await db.execute("DELETE FROM favorite_albums WHERE album_id = :a", {"a": album_id})
    await db.execute("DELETE FROM favorite_artists WHERE artist_id = :a", {"a": artist_id})
    await db.execute("DELETE FROM favorites WHERE track_id = :t", {"t": track_id})
    await db.execute("DELETE FROM tracks WHERE id = :t", {"t": track_id})
    await db.execute("DELETE FROM albums WHERE id = :a", {"a": album_id})
    await db.execute("DELETE FROM artists WHERE id = :a", {"a": artist_id})


@pytest.fixture
def offline_resolve(monkeypatch):
    """Keep detail endpoints local: no connector/network lookups in tests."""

    async def _none(*_args, **_kwargs):
        return None

    async def _false(*_args, **_kwargs):
        return False

    monkeypatch.setattr("app.services.resolve.find_album_online", _none)
    monkeypatch.setattr("app.services.resolve.find_artist_online", _none)
    monkeypatch.setattr("app.services.resolve.link_album_via_artist", _false)


@pytest.mark.asyncio
async def test_album_card_like_is_not_download_all(client, auth_headers, offline_resolve):
    track_id, artist_id, album_id = await _entities("CardAlbum")
    try:
        liked = await client.put(f"/v1/albums/{album_id}/like", headers=auth_headers)
        assert liked.status_code == 204, liked.text

        detail = await client.get(f"/v1/albums/{album_id}", headers=auth_headers)
        assert detail.status_code == 200, detail.text
        assert detail.json()["isFavorite"] is True

        library = await client.get("/v1/library/albums", headers=auth_headers)
        assert library.status_code == 200, library.text
        assert album_id in {int(a["id"]) for a in library.json()["items"]}

        # Card actions must not like the album's tracks (no downloads).
        user = await db.fetch_one("SELECT id FROM users ORDER BY id LIMIT 1")
        assert (
            await db.fetch_one(
                "SELECT 1 FROM favorites WHERE user_id = :u AND track_id = :t",
                {"u": user["id"], "t": track_id},
            )
            is None
        )

        removed = await client.delete(f"/v1/albums/{album_id}/like", headers=auth_headers)
        assert removed.status_code == 204
        detail = await client.get(f"/v1/albums/{album_id}", headers=auth_headers)
        assert detail.json()["isFavorite"] is False

        missing = await client.put("/v1/albums/999999999/like", headers=auth_headers)
        assert missing.status_code == 404
    finally:
        await _cleanup_entities(track_id, artist_id, album_id)


@pytest.mark.asyncio
async def test_artist_card_like_and_flag(client, auth_headers, offline_resolve):
    track_id, artist_id, album_id = await _entities("CardArtist")
    try:
        liked = await client.put(f"/v1/artists/{artist_id}/like", headers=auth_headers)
        assert liked.status_code == 204, liked.text

        detail = await client.get(f"/v1/artists/{artist_id}", headers=auth_headers)
        assert detail.status_code == 200, detail.text
        assert detail.json()["isFavorite"] is True

        removed = await client.delete(f"/v1/artists/{artist_id}/like", headers=auth_headers)
        assert removed.status_code == 204
        detail = await client.get(f"/v1/artists/{artist_id}", headers=auth_headers)
        assert detail.json()["isFavorite"] is False

        missing = await client.put("/v1/artists/999999999/like", headers=auth_headers)
        assert missing.status_code == 404
    finally:
        await _cleanup_entities(track_id, artist_id, album_id)


@pytest.mark.asyncio
async def test_library_playlists_include_followed(client, auth_headers):
    me = await db.fetch_one("SELECT id FROM users ORDER BY id LIMIT 1")
    other = await db.execute_returning(
        """
        INSERT INTO users (username, password_hash, display_name)
        VALUES ('library-other', 'x', 'Other') RETURNING id
        """
    )
    other_id = int(other[0]["id"])
    playlist = await playlists_service.create(other_id, "Followed Playlist", visibility="public")
    assert playlist is not None
    playlist_id = int(playlist["id"])
    try:
        before = await client.get("/v1/library/playlists", headers=auth_headers)
        assert playlist_id not in {int(p["id"]) for p in before.json()["items"]}

        followed = await client.put(
            f"/v1/playlists/{playlist_id}/follow", headers=auth_headers
        )
        assert followed.status_code == 204, followed.text

        after = await client.get("/v1/library/playlists", headers=auth_headers)
        item = next(
            (p for p in after.json()["items"] if int(p["id"]) == playlist_id), None
        )
        assert item is not None
        assert item["isFollowed"] is True

        assert me is not None
    finally:
        await db.execute("DELETE FROM playlist_follows WHERE playlist_id = :p", {"p": playlist_id})
        await db.execute("DELETE FROM playlists WHERE id = :p", {"p": playlist_id})
        await db.execute("DELETE FROM users WHERE id = :u", {"u": other_id})


@pytest.mark.asyncio
async def test_artist_tracks_lists_more_than_top_twenty(client, auth_headers, offline_resolve):
    track_id, artist_id, album_id = await _entities("AllTracks")
    extra_ids: list[int] = []
    try:
        for index in range(24):
            rows = await db.execute_returning(
                """
                INSERT INTO tracks (title, artists, album, duration_sec)
                VALUES (:t, ARRAY['AllTracks Artist'], 'AllTracks Album', 180)
                RETURNING id
                """,
                {"t": f"AllTracks Extra {index}"},
            )
            extra_ids.append(rows[0]["id"])

        response = await client.get(
            f"/v1/artists/{artist_id}/tracks?limit=50", headers=auth_headers
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["total"] >= 25
        assert len(body["items"]) >= 25
        titles = {item["title"] for item in body["items"]}
        assert "AllTracks Extra 0" in titles
    finally:
        await db.execute(
            "DELETE FROM tracks WHERE id = ANY(:ids)", {"ids": extra_ids + [track_id]}
        )
        await db.execute("DELETE FROM albums WHERE id = :a", {"a": album_id})
        await db.execute("DELETE FROM artists WHERE id = :a", {"a": artist_id})


@pytest.mark.asyncio
async def test_library_tracks_query_filters_by_title(client, auth_headers, offline_resolve):
    track_id, artist_id, album_id = await _entities("UniqueQuery")
    user = await db.fetch_one("SELECT id FROM users ORDER BY id LIMIT 1")
    try:
        await db.execute(
            "INSERT INTO favorites (user_id, track_id) VALUES (:u, :t)",
            {"u": user["id"], "t": track_id},
        )
        response = await client.get(
            "/v1/library/tracks?q=UniqueQuery", headers=auth_headers
        )
        assert response.status_code == 200, response.text
        assert any(item["id"] == str(track_id) for item in response.json()["items"])

        missing = await client.get(
            "/v1/library/tracks?q=NoSuchTrackXYZ", headers=auth_headers
        )
        assert missing.status_code == 200
        assert missing.json()["items"] == []
    finally:
        await db.execute("DELETE FROM favorites WHERE track_id = :t", {"t": track_id})
        await _cleanup_entities(track_id, artist_id, album_id)


@pytest.mark.asyncio
async def test_playlist_cover_upload_and_delete(client, auth_headers):
    import io

    from PIL import Image

    created = await client.post(
        "/v1/playlists",
        headers=auth_headers,
        json={"title": "Cover Test", "visibility": "private"},
    )
    assert created.status_code == 201, created.text
    playlist_id = created.json()["id"]
    assert created.json()["cover"]["id"] == "placeholder"
    try:
        buf = io.BytesIO()
        Image.new("RGB", (64, 64), (40, 80, 120)).save(buf, format="PNG")
        uploaded = await client.post(
            f"/v1/playlists/{playlist_id}/cover",
            headers=auth_headers,
            files={"file": ("cover.png", buf.getvalue(), "image/png")},
        )
        assert uploaded.status_code == 200, uploaded.text
        cover_id = uploaded.json()["cover"]["id"]
        assert cover_id != "placeholder"

        detail = await client.get(f"/v1/playlists/{playlist_id}", headers=auth_headers)
        assert detail.json()["cover"]["id"] == cover_id

        image = await client.get(f"/v1/images/{cover_id}?size=sm", headers=auth_headers)
        assert image.status_code == 200
        assert image.headers["content-type"].startswith("image/")

        cleared = await client.delete(
            f"/v1/playlists/{playlist_id}/cover", headers=auth_headers
        )
        assert cleared.status_code == 204
        detail = await client.get(f"/v1/playlists/{playlist_id}", headers=auth_headers)
        assert detail.json()["cover"]["id"] == "placeholder"
    finally:
        await client.delete(f"/v1/playlists/{playlist_id}", headers=auth_headers)


@pytest.mark.asyncio
async def test_weekly_releases_rebuilds_same_playlist(monkeypatch):
    from datetime import date

    from app.services import weekly_releases

    uid = (
        await db.execute_returning(
            "INSERT INTO users (username, password_hash, display_name) "
            "VALUES ('weekly-user', 'x', 'Weekly') RETURNING id"
        )
    )[0]["id"]
    artist = await db.execute_returning(
        "INSERT INTO artists (name, last_seen_at) VALUES ('Weekly Artist', NOW()) RETURNING id"
    )
    artist_id = artist[0]["id"]
    album = await db.execute_returning(
        """
        INSERT INTO albums (title, artist_id, release_date, last_seen_at)
        VALUES ('Weekly Album', :a, :d, NOW()) RETURNING id
        """,
        {"a": artist_id, "d": date.today()},
    )
    album_id = album[0]["id"]
    track = await db.execute_returning(
        """
        INSERT INTO tracks (title, artists, album, duration_sec, last_seen_at)
        VALUES ('Weekly Track', ARRAY['Weekly Artist'], 'Weekly Album', 180, NOW())
        RETURNING id
        """
    )
    track_id = track[0]["id"]
    await db.execute(
        "INSERT INTO favorite_artists (user_id, artist_id) VALUES (:u, :a)",
        {"u": uid, "a": artist_id},
    )

    async def fake_tracks(album_id: int, user_id: int) -> list[int]:
        return [track_id]

    monkeypatch.setattr(weekly_releases, "_album_track_ids", fake_tracks)

    playlist_id = None
    try:
        first = await weekly_releases.run(force=True)
        assert first["users"] >= 1
        state = await db.fetch_one(
            "SELECT playlist_id FROM weekly_release_playlists WHERE user_id = :u",
            {"u": uid},
        )
        assert state is not None
        playlist_id = int(state["playlist_id"])
        rows = await db.fetch_all(
            "SELECT track_id FROM playlist_tracks WHERE playlist_id = :p",
            {"p": playlist_id},
        )
        assert [int(r["track_id"]) for r in rows] == [track_id]

        # Same playlist id on the next run, and stale tracks are removed.
        await db.execute(
            "UPDATE albums SET release_date = NOW() - INTERVAL '30 days' WHERE id = :a",
            {"a": album_id},
        )
        await weekly_releases.run(force=True)
        state2 = await db.fetch_one(
            "SELECT playlist_id FROM weekly_release_playlists WHERE user_id = :u",
            {"u": uid},
        )
        assert int(state2["playlist_id"]) == playlist_id
        rows = await db.fetch_all(
            "SELECT track_id FROM playlist_tracks WHERE playlist_id = :p",
            {"p": playlist_id},
        )
        assert rows == []
    finally:
        if playlist_id:
            await db.execute("DELETE FROM weekly_release_playlists WHERE user_id = :u", {"u": uid})
            await db.execute("DELETE FROM playlists WHERE id = :p", {"p": playlist_id})
        await db.execute("DELETE FROM favorite_artists WHERE artist_id = :a", {"a": artist_id})
        await db.execute("DELETE FROM tracks WHERE id = :t", {"t": track_id})
        await db.execute("DELETE FROM albums WHERE id = :a", {"a": album_id})
        await db.execute("DELETE FROM artists WHERE id = :a", {"a": artist_id})
        await db.execute("DELETE FROM users WHERE id = :u", {"u": uid})


@pytest.mark.asyncio
async def test_artist_latest_release_and_sync_status(client, auth_headers, offline_resolve):
    track_id, artist_id, album_id = await _entities("LatestRelease")
    try:
        await db.execute(
            "UPDATE albums SET release_date = DATE '2026-11-01' WHERE id = :a",
            {"a": album_id},
        )
        detail = await client.get(f"/v1/artists/{artist_id}", headers=auth_headers)
        assert detail.status_code == 200, detail.text
        body = detail.json()
        assert body["latestRelease"]["id"] == str(album_id)
        assert body["latestRelease"]["releaseDate"] == "2026-11-01"

        status = await client.get(f"/v1/artists/{artist_id}/sync", headers=auth_headers)
        assert status.status_code == 200
        assert status.json()["status"] == "idle"
        assert isinstance(status.json()["artistId"], str)
    finally:
        await _cleanup_entities(track_id, artist_id, album_id)
