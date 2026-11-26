"""Favorites lifecycle tests: like, unlike, grace, re-like (needs Postgres)."""

import pytest

# tests run inside the api container where DATABASE_URL points to test db
from app import db
from app.auth import ensure_seed_user
from app.services import favorites


@pytest.fixture(autouse=True)
async def _schema():
    await db.apply_schema()
    await ensure_seed_user()
    yield
    await db.close_engine()


@pytest.mark.asyncio
async def test_like_unlike_cycle():
    await ensure_seed_user()
    user = await db.fetch_one("SELECT id FROM users LIMIT 1")
    uid = user["id"]

    # create an ephemeral track manually
    rows = await db.execute_returning(
        "INSERT INTO tracks (title, artists, album, duration_sec) VALUES ('T', '{A}', '', 200) RETURNING id"
    )
    tid = rows[0]["id"]

    assert (await favorites.like(uid, tid))["ok"]
    favs = await favorites.list_favorites(uid)
    assert any(t["id"] == tid for t in favs)

    assert (await favorites.unlike(uid, tid))["ok"]
    favs = await favorites.list_favorites(uid)
    assert not any(t["id"] == tid for t in favs)

    # unlike-grace marker set on local sources only; remote sources untouched
    src = await db.fetch_one(
        "SELECT pending_delete_since FROM track_sources WHERE track_id = :t AND connector = 'local'",
        {"t": tid},
    )
    assert src is None  # no local source -> no grace row
    # cleanup
    await db.execute("DELETE FROM tracks WHERE id = :t", {"t": tid})


@pytest.mark.asyncio
async def test_unlike_sets_grace_only_when_alone():
    await ensure_seed_user()
    users = await db.fetch_all("SELECT id FROM users")
    u1 = users[0]["id"]
    rows = await db.execute_returning(
        "INSERT INTO tracks (title, artists, album, duration_sec) VALUES ('T2', '{A}', '', 200) RETURNING id"
    )
    tid = rows[0]["id"]
    # local source simulating a downloaded file
    await db.execute(
        "INSERT INTO track_sources (track_id, connector, external_id, file_path) VALUES (:t, 'local', 'x', '/tmp/x.mp3')",
        {"t": tid},
    )
    await favorites.like(u1, tid)
    await favorites.unlike(u1, tid)
    row = await db.fetch_one(
        "SELECT pending_delete_since FROM track_sources WHERE external_id = 'x'"
    )
    assert row["pending_delete_since"] is not None
    await favorites.like(u1, tid)
    row = await db.fetch_one(
        "SELECT pending_delete_since FROM track_sources WHERE external_id = 'x'"
    )
    assert row["pending_delete_since"] is None
    await db.execute("DELETE FROM tracks WHERE id = :t", {"t": tid})
