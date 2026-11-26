"""Track identity: one recording = one like state, one library entry.

Regression suite for the «лайк отваливается / дубли в библиотеке» bug: the
same recording arrives as several tracks rows (collab packed differently,
duration ±15 s), likes used to live strictly on one row id.
"""

import uuid

import pytest

from app import db
from app.services.identity import identity_hash, primary_artist_key, same_recording


@pytest.fixture(autouse=True)
async def _schema():
    await db.apply_schema()
    yield
    await db.close_engine()


# ── pure identity rules ──────────────────────────────────────────────────────


def test_identity_folds_collab_packing():
    # ["Тимати","MOT"] vs ["Тимати,MOT,ЕГОР КРИД"] share the primary artist
    assert identity_hash("Song", ["Тимати", "MOT"]) == identity_hash("Song", ["Тимати,MOT,ЕГОР КРИД"])
    assert primary_artist_key(["Тимати,MOT,ЕГОР КРИД"]) == primary_artist_key(["Тимати", "MOT"])


def test_identity_keeps_different_recordings_apart():
    assert identity_hash("Song", ["A"]) != identity_hash("Song", ["B"])
    assert identity_hash("Song", ["A"]) != identity_hash("Other", ["A"])
    # duration matters at comparison time, not in the hash
    assert same_recording("Song", ["A"], 200, "Song", ["A"], 214)
    assert not same_recording("Song", ["A"], 200, "Song", ["A"], 260)


def test_monsta_x_stays_whole():
    assert primary_artist_key(["MONSTA X"]) == primary_artist_key(["MONSTA X"])
    from app.services.normalize import flatten_artist_names, split_artists_strict

    assert flatten_artist_names(["MONSTA X"]) == ["MONSTA X"]
    assert split_artists_strict("MONSTA X") == ["MONSTA X"]
    assert split_artists_strict("A feat. B & C") == ["A", "B", "C"]


# ── cluster-aware likes ──────────────────────────────────────────────────────


async def _mk_track(title: str, artists: list[str], dur: int) -> int:
    rows = await db.execute_returning(
        "INSERT INTO tracks (title, artists, album, duration_sec, identity_hash, last_seen_at) "
        "VALUES (:t, :a, '', :d, :h, NOW()) RETURNING id",
        {"t": title, "a": artists, "d": dur, "h": identity_hash(title, artists)},
    )
    return rows[0]["id"]


async def _mk_user() -> int:
    s = uuid.uuid4().hex[:8]
    rows = await db.execute_returning(
        "INSERT INTO users (username, password_hash, email, role) VALUES (:u, 'x', :e, 'user') RETURNING id",
        {"u": f"id-{s}", "e": f"id-{s}@example.com"},
    )
    return rows[0]["id"]


async def _cleanup(uid: int, *tids: int):
    await db.execute("DELETE FROM favorites WHERE user_id = :u", {"u": uid})
    await db.execute("DELETE FROM users WHERE id = :u", {"u": uid})
    for t in tids:
        await db.execute("DELETE FROM favorites WHERE track_id = :t", {"t": t})
        await db.execute("DELETE FROM tracks WHERE id = :t", {"t": t})


@pytest.mark.asyncio
async def test_like_on_twin_shows_liked_on_both_rows():
    from app.services.dto import build_track_dtos

    a = await _mk_track("Song", ["XOLIDAYBOY"], 200)
    b = await _mk_track("Song", ["XOLIDAYBOY"], 210)  # twin: ±15 s window
    uid = await _mk_user()
    await db.execute("INSERT INTO favorites (user_id, track_id) VALUES (:u, :t)", {"u": uid, "t": a})

    dtos = await build_track_dtos([a, b], uid)
    assert all(d["is_favorite"] for d in dtos), "лайк на кластере должен быть виден на обеих строках"

    await _cleanup(uid, a, b)


@pytest.mark.asyncio
async def test_unlike_clears_whole_cluster():
    from app.services import favorites

    a = await _mk_track("Song2", ["A"], 100)
    b = await _mk_track("Song2", ["A"], 108)
    uid = await _mk_user()
    await favorites.like(uid, a)
    await db.execute("INSERT INTO favorites (user_id, track_id) VALUES (:u, :t)", {"u": uid, "t": b})

    await favorites.unlike(uid, a)
    n = await db.fetch_one(
        "SELECT COUNT(*) AS n FROM favorites WHERE user_id = :u", {"u": uid}
    )
    assert int(n["n"]) == 0, "unlike должен снимать лайк со всех строк кластера"

    await _cleanup(uid, a, b)


@pytest.mark.asyncio
async def test_like_twice_on_twins_keeps_single_library_row():
    from app.services import favorites

    a = await _mk_track("Song3", ["A"], 100)
    b = await _mk_track("Song3", ["A"], 110)
    uid = await _mk_user()
    await favorites.like(uid, a)
    await favorites.like(uid, b)  # user likes the twin from search

    n = await db.fetch_one(
        "SELECT COUNT(*) AS n FROM favorites WHERE user_id = :u", {"u": uid}
    )
    assert int(n["n"]) == 1, "повторный лайк двойника не должен плодить дубли в библиотеке"

    await _cleanup(uid, a, b)


@pytest.mark.asyncio
async def test_repair_backfills_identity_hash():
    from app.repair import backfill_identity_hash

    rows = await db.execute_returning(
        "INSERT INTO tracks (title, artists, album, duration_sec, last_seen_at) "
        "VALUES ('Backfill Me', ARRAY['A'], '', 100, NOW()) RETURNING id"
    )
    tid = rows[0]["id"]
    res = await backfill_identity_hash()
    assert res["filled"] >= 1
    row = await db.fetch_one("SELECT identity_hash FROM tracks WHERE id = :i", {"i": tid})
    assert row["identity_hash"] == identity_hash("Backfill Me", ["A"])
    await db.execute("DELETE FROM tracks WHERE id = :i", {"i": tid})
