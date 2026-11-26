"""Entity favorites: album/artist likes, listing, entity materialisation, GC pins.

Runs against the compose Postgres (same style as test_favorites.py): creates
ephemeral rows and cleans them up afterwards.
"""

import pytest

# tests run inside the api container where DATABASE_URL points to the db
from app import db
from app.auth import ensure_seed_user
from app.services import favorites
from app.services.albums import list_favorite_albums
from app.services.artists import list_favorite_artists
from app.services.entity_upsert import ensure_entities_for_track
from app.services.gc import run_gc


@pytest.fixture(autouse=True)
async def _schema():
    await db.apply_schema()
    await ensure_seed_user()
    yield
    await db.close_engine()


async def _user_id() -> int:
    await ensure_seed_user()
    user = await db.fetch_one("SELECT id FROM users LIMIT 1")
    return user["id"]


async def _make_track_with_entities(prefix: str) -> tuple[int, int, int]:
    """Return (track_id, artist_id, album_id) with materialised entities."""
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
    return track_id, artist["id"], album["id"]


async def _cleanup(track_id: int, artist_id: int, album_id: int) -> None:
    await db.execute("DELETE FROM favorite_albums WHERE album_id = :a", {"a": album_id})
    await db.execute("DELETE FROM favorite_artists WHERE artist_id = :a", {"a": artist_id})
    await db.execute("DELETE FROM favorites WHERE track_id = :t", {"t": track_id})
    await db.execute("DELETE FROM tracks WHERE id = :t", {"t": track_id})
    await db.execute("DELETE FROM albums WHERE id = :a", {"a": album_id})
    await db.execute("DELETE FROM artists WHERE id = :a", {"a": artist_id})


async def _age(prefix: str, days: int = 40) -> None:
    await db.execute(
        "UPDATE artists SET last_seen_at = NOW() - make_interval(days => :d) WHERE name = :n",
        {"d": days, "n": f"{prefix} Artist"},
    )
    await db.execute(
        "UPDATE albums SET last_seen_at = NOW() - make_interval(days => :d) WHERE title = :t",
        {"d": days, "t": f"{prefix} Album"},
    )
    await db.execute(
        "UPDATE tracks SET last_seen_at = NOW() - make_interval(days => :d) WHERE title = :t",
        {"d": days, "t": f"{prefix} Track"},
    )


@pytest.mark.asyncio
async def test_ensure_entities_materialises_rows():
    uid = await _user_id()
    track_id, artist_id, album_id = await _make_track_with_entities("Ensure")
    try:
        assert artist_id > 0 and album_id > 0
        # idempotent
        await ensure_entities_for_track(track_id)
        rows = await db.fetch_all(
            "SELECT id FROM artists WHERE name = 'Ensure Artist'"
        )
        assert len(rows) == 1
    finally:
        await _cleanup(track_id, artist_id, album_id)
        assert uid > 0


@pytest.mark.asyncio
async def test_album_like_unlike_and_listing():
    uid = await _user_id()
    track_id, artist_id, album_id = await _make_track_with_entities("AlbumLike")
    try:
        assert (await favorites.like_album(uid, album_id))["ok"]
        page = await list_favorite_albums(uid, 200, 0)
        assert any(a["id"] == album_id for a in page["items"])

        # idempotent like
        await favorites.like_album(uid, album_id)
        page = await list_favorite_albums(uid, 200, 0)
        assert sum(1 for a in page["items"] if a["id"] == album_id) == 1

        assert not (await favorites.like_album(uid, 10**9))["ok"]

        await favorites.unlike_album(uid, album_id)
        page = await list_favorite_albums(uid, 200, 0)
        assert not any(a["id"] == album_id for a in page["items"])
    finally:
        await _cleanup(track_id, artist_id, album_id)


@pytest.mark.asyncio
async def test_artist_like_unlike_and_listing():
    uid = await _user_id()
    track_id, artist_id, album_id = await _make_track_with_entities("ArtistLike")
    try:
        assert (await favorites.like_artist(uid, artist_id))["ok"]
        page = await list_favorite_artists(uid, 200, 0)
        assert any(a["id"] == artist_id for a in page["items"])

        await favorites.like_artist(uid, artist_id)
        page = await list_favorite_artists(uid, 200, 0)
        assert sum(1 for a in page["items"] if a["id"] == artist_id) == 1

        assert not (await favorites.like_artist(uid, 10**9))["ok"]

        await favorites.unlike_artist(uid, artist_id)
        page = await list_favorite_artists(uid, 200, 0)
        assert not any(a["id"] == artist_id for a in page["items"])
    finally:
        await _cleanup(track_id, artist_id, album_id)


@pytest.mark.asyncio
async def test_liking_track_materialises_entities():
    uid = await _user_id()
    rows = await db.execute_returning(
        """
        INSERT INTO tracks (title, artists, album, duration_sec)
        VALUES ('NavTest Track', ARRAY['NavTest Artist'], 'NavTest Album', 120)
        RETURNING id
        """,
    )
    track_id = rows[0]["id"]
    artist_id = 0
    album_id = 0
    try:
        assert (await favorites.like(uid, track_id))["ok"]
        artist = await db.fetch_one("SELECT id FROM artists WHERE name = 'NavTest Artist'")
        album = await db.fetch_one("SELECT id FROM albums WHERE title = 'NavTest Album'")
        assert artist is not None, "liking a track must create its artist row"
        assert album is not None, "liking a track must create its album row"
        artist_id, album_id = artist["id"], album["id"]
        await favorites.unlike(uid, track_id)
    finally:
        await _cleanup(track_id, artist_id, album_id)


@pytest.mark.asyncio
async def test_gc_pins_liked_entities_and_sweeps_the_rest():
    uid = await _user_id()
    liked_track, liked_artist, liked_album = await _make_track_with_entities("PinKeep")
    gone_track, gone_artist, gone_album = await _make_track_with_entities("PinDrop")
    try:
        await _age("PinKeep")
        await _age("PinDrop")
        assert (await favorites.like_album(uid, liked_album))["ok"]
        assert (await favorites.like_artist(uid, liked_artist))["ok"]

        await run_gc()

        assert await db.fetch_one("SELECT id FROM artists WHERE id = :i", {"i": liked_artist})
        assert await db.fetch_one("SELECT id FROM albums WHERE id = :i", {"i": liked_album})
        assert await db.fetch_one("SELECT id FROM tracks WHERE id = :i", {"i": liked_track})

        assert await db.fetch_one("SELECT id FROM artists WHERE id = :i", {"i": gone_artist}) is None
        assert await db.fetch_one("SELECT id FROM albums WHERE id = :i", {"i": gone_album}) is None
        assert await db.fetch_one("SELECT id FROM tracks WHERE id = :i", {"i": gone_track}) is None
    finally:
        await _cleanup(liked_track, liked_artist, liked_album)
        await _cleanup(gone_track, gone_artist, gone_album)
