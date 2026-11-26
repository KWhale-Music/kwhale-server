"""GC tests: playlist tracks are pinned from the ephemeral sweep (data-loss fix).

Before the fix, a track that lived only in a user's playlist (no like, no local
file) was deleted after the ephemeral TTL and silently vanished from the
playlist via ON DELETE CASCADE. These tests pin that behaviour down.
"""

import uuid

import pytest

from app import db
from app.services.gc import run_gc


@pytest.fixture(autouse=True)
async def _schema():
    await db.apply_schema()
    yield
    await db.close_engine()


async def _mk_old_track(title: str) -> int:
    rows = await db.execute_returning(
        "INSERT INTO tracks (title, artists, album, duration_sec, last_seen_at) "
        "VALUES (:t, '{Test Artist}', '', 10, NOW() - INTERVAL '30 days') RETURNING id",
        {"t": title},
    )
    return rows[0]["id"]


async def _mk_user() -> int:
    suffix = uuid.uuid4().hex[:8]
    rows = await db.execute_returning(
        "INSERT INTO users (username, password_hash, email, role) "
        "VALUES (:u, 'x', :e, 'user') RETURNING id",
        {"u": f"gc-{suffix}", "e": f"gc-{suffix}@example.com"},
    )
    return rows[0]["id"]


@pytest.mark.asyncio
async def test_playlist_track_survives_ephemeral_sweep():
    tid = await _mk_old_track("Playlist-only track")
    uid = await _mk_user()
    pid = (
        await db.execute_returning(
            "INSERT INTO playlists (user_id, name) VALUES (:u, 'GC playlist') RETURNING id",
            {"u": uid},
        )
    )[0]["id"]
    await db.execute(
        "INSERT INTO playlist_tracks (playlist_id, track_id, position) VALUES (:p, :t, 0)",
        {"p": pid, "t": tid},
    )

    await run_gc()

    still = await db.fetch_one("SELECT 1 FROM tracks WHERE id = :t", {"t": tid})
    assert still is not None, "трек из плейлиста НЕ должен удаляться GC (playlist pin)"

    await db.execute("DELETE FROM playlists WHERE id = :p", {"p": pid})
    await db.execute("DELETE FROM users WHERE id = :u", {"u": uid})
    await db.execute("DELETE FROM tracks WHERE id = :t", {"t": tid})


@pytest.mark.asyncio
async def test_unpinned_old_track_is_swept():
    tid = await _mk_old_track("Ephemeral track")

    stats = await run_gc()

    gone = await db.fetch_one("SELECT 1 FROM tracks WHERE id = :t", {"t": tid})
    assert gone is None, "незакреплённый старый трек должен удаляться GC"
    assert stats["tracks_deleted"] >= 1


async def _mk_image(url: str, *, days_old: int = 30) -> str:
    image_id = f"img-{uuid.uuid4().hex}"
    await db.execute(
        """
        INSERT INTO images
            (id, source_url, mime, data, fetched_at, created_at, last_accessed_at)
        VALUES
            (:i, :u, 'image/jpeg', :dta, NOW() - make_interval(days => :d),
             NOW() - make_interval(days => :d), NOW() - make_interval(days => :d))
        """,
        {"i": image_id, "u": url, "d": days_old, "dta": b"x"},
    )
    return image_id


@pytest.mark.asyncio
async def test_unpinned_search_image_expires_after_ttl():
    image_id = await _mk_image("https://cdn.example/search-cover.jpg")
    await db.execute(
        "INSERT INTO image_variants (image_id, size, mime, data, etag) "
        "VALUES (:i, 'md', 'image/jpeg', :dta, 'e')",
        {"i": image_id, "dta": b"v"},
    )

    stats = await run_gc()

    assert await db.fetch_one("SELECT 1 FROM images WHERE id = :i", {"i": image_id}) is None
    assert await db.fetch_one(
        "SELECT 1 FROM image_variants WHERE image_id = :i", {"i": image_id}
    ) is None
    assert stats["images_deleted"] >= 1


@pytest.mark.asyncio
async def test_liked_track_cover_is_pinned_forever():
    uid = await _mk_user()
    track = await db.execute_returning(
        """
        INSERT INTO tracks (title, artists, album, duration_sec, cover_url, last_seen_at)
        VALUES ('Pinned cover track', '{Artist}', '', 10,
                'https://cdn.example/liked-cover.jpg', NOW())
        RETURNING id
        """
    )
    tid = track[0]["id"]
    await db.execute(
        "INSERT INTO favorites (user_id, track_id) VALUES (:u, :t)",
        {"u": uid, "t": tid},
    )
    image_id = await _mk_image("https://cdn.example/liked-cover.jpg")

    await run_gc()

    assert await db.fetch_one("SELECT 1 FROM images WHERE id = :i", {"i": image_id}) is not None

    await db.execute("DELETE FROM favorites WHERE track_id = :t", {"t": tid})
    await db.execute("DELETE FROM users WHERE id = :u", {"u": uid})
    await db.execute("DELETE FROM tracks WHERE id = :t", {"t": tid})
    await db.execute("DELETE FROM images WHERE id = :i", {"i": image_id})


@pytest.mark.asyncio
async def test_fresh_search_image_survives_until_ttl():
    image_id = await _mk_image("https://cdn.example/fresh-cover.jpg", days_old=1)
    await run_gc()
    assert await db.fetch_one("SELECT 1 FROM images WHERE id = :i", {"i": image_id}) is not None
    await db.execute("DELETE FROM images WHERE id = :i", {"i": image_id})


@pytest.mark.asyncio
async def test_expired_search_cache_rows_are_purged():
    from app.services.gc import purge_search_cache

    await db.execute(
        """
        INSERT INTO search_cache (cache_key, connector, result_json, expires_at)
        VALUES ('expired-key', 'test', '{}', NOW() - INTERVAL '1 hour'),
               ('fresh-key', 'test', '{}', NOW() + INTERVAL '1 hour')
        ON CONFLICT (cache_key) DO UPDATE SET expires_at = EXCLUDED.expires_at
        """
    )
    removed = await purge_search_cache()
    assert removed >= 1
    assert await db.fetch_one(
        "SELECT 1 FROM search_cache WHERE cache_key = 'expired-key'"
    ) is None
    assert await db.fetch_one(
        "SELECT 1 FROM search_cache WHERE cache_key = 'fresh-key'"
    ) is not None
    await db.execute("DELETE FROM search_cache WHERE cache_key = 'fresh-key'")


@pytest.mark.asyncio
async def test_liked_artist_pins_their_tracks():
    uid = await _mk_user()
    artist = await db.execute_returning(
        "INSERT INTO artists (name, last_seen_at) VALUES ('GC Liked Artist', NOW() - INTERVAL '40 days') RETURNING id"
    )
    artist_id = artist[0]["id"]
    track = await db.execute_returning(
        """
        INSERT INTO tracks (title, artists, album, duration_sec, last_seen_at)
        VALUES ('Pinned artist track', ARRAY['GC Liked Artist'], '', 10,
                NOW() - INTERVAL '40 days')
        RETURNING id
        """
    )
    track_id = track[0]["id"]
    await db.execute(
        "INSERT INTO favorite_artists (user_id, artist_id) VALUES (:u, :a)",
        {"u": uid, "a": artist_id},
    )

    await run_gc()

    assert await db.fetch_one("SELECT 1 FROM tracks WHERE id = :t", {"t": track_id}) is not None
    assert await db.fetch_one("SELECT 1 FROM artists WHERE id = :a", {"a": artist_id}) is not None

    await db.execute("DELETE FROM favorite_artists WHERE artist_id = :a", {"a": artist_id})
    await db.execute("DELETE FROM users WHERE id = :u", {"u": uid})
    await db.execute("DELETE FROM tracks WHERE id = :t", {"t": track_id})
    await db.execute("DELETE FROM artists WHERE id = :a", {"a": artist_id})


@pytest.mark.asyncio
async def test_liked_album_and_artist_covers_are_pinned():
    uid = await _mk_user()
    artist = await db.execute_returning(
        "INSERT INTO artists (name, cover_url, last_seen_at) "
        "VALUES ('GC Cover Artist', 'https://cdn.example/artist.jpg', NOW()) RETURNING id"
    )
    artist_id = artist[0]["id"]
    album = await db.execute_returning(
        "INSERT INTO albums (title, artist_id, cover_url, last_seen_at) "
        "VALUES ('GC Cover Album', :a, 'https://cdn.example/album.jpg', NOW()) RETURNING id",
        {"a": artist_id},
    )
    album_id = album[0]["id"]
    await db.execute(
        "INSERT INTO favorite_artists (user_id, artist_id) VALUES (:u, :a)",
        {"u": uid, "a": artist_id},
    )
    await db.execute(
        "INSERT INTO favorite_albums (user_id, album_id) VALUES (:u, :a)",
        {"u": uid, "a": album_id},
    )
    artist_image = await _mk_image("https://cdn.example/artist.jpg")
    album_image = await _mk_image("https://cdn.example/album.jpg")
    try:
        await run_gc()
        assert await db.fetch_one(
            "SELECT 1 FROM images WHERE id = :i", {"i": artist_image}
        ) is not None
        assert await db.fetch_one(
            "SELECT 1 FROM images WHERE id = :i", {"i": album_image}
        ) is not None
    finally:
        await db.execute("DELETE FROM favorite_albums WHERE album_id = :a", {"a": album_id})
        await db.execute("DELETE FROM favorite_artists WHERE artist_id = :a", {"a": artist_id})
        await db.execute("DELETE FROM users WHERE id = :u", {"u": uid})
        await db.execute("DELETE FROM albums WHERE id = :a", {"a": album_id})
        await db.execute("DELETE FROM artists WHERE id = :a", {"a": artist_id})
        await db.execute("DELETE FROM images WHERE id = ANY(:ids)", {"ids": [artist_image, album_image]})
