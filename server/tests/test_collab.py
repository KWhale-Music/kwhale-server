"""Collab artist splitting: participants become real rows, band names stay whole.

The evidence-gated splitter must shred "Тимати,MOT,ЕГОР КРИД" (corroborated
by another source or existing rows) but never shred "Tyler, The Creator" /
"MONSTA X" without support.
"""

import uuid

import pytest

from app import db
from app.services.entity_upsert import resolve_artist_names
from app.services.merge import split_collab_artists
from app.services.normalize import flatten_artist_names


@pytest.fixture(autouse=True)
async def _schema():
    await db.apply_schema()
    yield
    await db.close_engine()


async def _mk_artist(name: str) -> int:
    rows = await db.execute_returning(
        "INSERT INTO artists (name) VALUES (:n) RETURNING id", {"n": name}
    )
    return rows[0]["id"]


async def _mk_user() -> int:
    s = uuid.uuid4().hex[:8]
    rows = await db.execute_returning(
        "INSERT INTO users (username, password_hash, email, role) VALUES (:u, 'x', :e, 'user') RETURNING id",
        {"u": f"cl-{s}", "e": f"cl-{s}@example.com"},
    )
    return rows[0]["id"]


@pytest.mark.asyncio
async def test_split_when_another_source_sends_parts():
    resolved = await resolve_artist_names(
        ["Тимати,MOT,ЕГОР КРИД"],
        corroborating=[["Тимати", "MOT", "ЕГОР КРИД"]],
    )
    assert resolved == ["Тимати", "MOT", "ЕГОР КРИД"]


@pytest.mark.asyncio
async def test_split_when_parts_exist_as_artists():
    a = await _mk_artist("MOT")
    b = await _mk_artist("ЕГОР КРИД")
    resolved = await resolve_artist_names(["MOT, ЕГОР КРИД"])
    assert resolved == ["MOT", "ЕГОР КРИД"]
    await db.execute("DELETE FROM artists WHERE id IN (:a, :b)", {"a": a, "b": b})


@pytest.mark.asyncio
async def test_band_names_stay_whole_without_evidence():
    assert await resolve_artist_names(["Tyler, The Creator"]) == ["Tyler, The Creator"]
    assert await resolve_artist_names(["Simon & Garfunkel"]) == ["Simon & Garfunkel"]
    assert flatten_artist_names(["MONSTA X"]) == ["MONSTA X"]


@pytest.mark.asyncio
async def test_split_collab_artist_repoints_links():
    shell = await _mk_artist("Paul Anka, Elvis Presley, Dean Martin")
    p1 = await _mk_artist("Paul Anka")
    p2 = await _mk_artist("Elvis Presley")
    p3 = await _mk_artist("Dean Martin")
    uid = await _mk_user()
    t = await db.execute_returning(
        "INSERT INTO tracks (title, artists, album, duration_sec, last_seen_at) "
        "VALUES ('Duet', ARRAY['Paul Anka, Elvis Presley, Dean Martin'], '', 120, NOW()) RETURNING id"
    )
    tid = t[0]["id"]
    await db.execute(
        "INSERT INTO track_artists (track_id, artist_id) VALUES (:t, :s)", {"t": tid, "s": shell}
    )
    await db.execute(
        "INSERT INTO favorite_artists (user_id, artist_id) VALUES (:u, :s)", {"u": uid, "s": shell}
    )

    res = await split_collab_artists()
    assert res["split"] >= 1

    left = await db.fetch_one("SELECT id FROM artists WHERE id = :i", {"i": shell})
    assert left is None, "колляб-оболочка должна быть удалена"
    links = await db.fetch_all(
        "SELECT artist_id FROM track_artists WHERE track_id = :t", {"t": tid}
    )
    linked = {int(x["artist_id"]) for x in links}
    assert {p1, p2, p3} & linked, "трек должен быть привязан к участникам"
    fav = await db.fetch_one(
        "SELECT artist_id FROM favorite_artists WHERE user_id = :u", {"u": uid}
    )
    assert fav is not None and int(fav["artist_id"]) == p1, "лайк артиста уезжает на основного участника"

    await db.execute("DELETE FROM favorite_artists WHERE user_id = :u", {"u": uid})
    await db.execute("DELETE FROM users WHERE id = :u", {"u": uid})
    await db.execute("DELETE FROM track_artists WHERE track_id = :t", {"t": tid})
    await db.execute("DELETE FROM tracks WHERE id = :t", {"t": tid})
    for pid in (p1, p2, p3):
        await db.execute("DELETE FROM artists WHERE id = :i", {"i": pid})


@pytest.mark.asyncio
async def test_split_dry_run_touches_nothing():
    shell = await _mk_artist("dryrun, collab, shell")
    res = await split_collab_artists(dry_run=True)
    left = await db.fetch_one("SELECT id FROM artists WHERE id = :i", {"i": shell})
    assert left is not None
    await db.execute("DELETE FROM artists WHERE id = :i", {"i": shell})
    assert res["dry_run"] is True


@pytest.mark.asyncio
async def test_ensure_entities_links_every_participant():
    from app.services.entity_upsert import ensure_entities_for_track

    rows = await db.execute_returning(
        "INSERT INTO tracks (title, artists, album, duration_sec, last_seen_at) "
        "VALUES ('Collab Song', ARRAY['Alpha feat. Beta'], '', 100, NOW()) RETURNING id"
    )
    tid = rows[0]["id"]
    await ensure_entities_for_track(tid)
    links = await db.fetch_all(
        """
        SELECT ar.name FROM track_artists ta JOIN artists ar ON ar.id = ta.artist_id
        WHERE ta.track_id = :t
        """,
        {"t": tid},
    )
    names = {x["name"] for x in links}
    assert "Alpha" in names and "Beta" in names, f"оба участника должны быть привязаны, got {names}"

    await db.execute("DELETE FROM track_artists WHERE track_id = :t", {"t": tid})
    await db.execute("DELETE FROM artists WHERE name IN ('Alpha', 'Beta')")
    await db.execute("DELETE FROM tracks WHERE id = :t", {"t": tid})
