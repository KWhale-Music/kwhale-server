"""Duplicate-merge tests: same recording with noisy durations is consolidated,
references are preserved, and distinct covers/versions are never merged."""

import uuid

import pytest

from app import db
from app.services.merge import merge_duplicates


@pytest.fixture(autouse=True)
async def _schema():
    await db.apply_schema()
    yield
    await db.close_engine()


async def _mk_track(title: str, artists: list[str], dur: int) -> int:
    rows = await db.execute_returning(
        "INSERT INTO tracks (title, artists, album, duration_sec, last_seen_at) "
        "VALUES (:t, :a, '', :d, NOW()) RETURNING id",
        {"t": title, "a": artists, "d": dur},
    )
    return rows[0]["id"]


async def _mk_user() -> int:
    s = uuid.uuid4().hex[:8]
    rows = await db.execute_returning(
        "INSERT INTO users (username, password_hash, email, role) VALUES (:u, 'x', :e, 'user') RETURNING id",
        {"u": f"m-{s}", "e": f"m-{s}@example.com"},
    )
    return rows[0]["id"]


@pytest.mark.asyncio
async def test_merges_same_recording_preserves_references():
    a = await _mk_track("Группа крови", ["Кино"], 240)
    b = await _mk_track("Группа крови", ["Кино"], 247)  # noisy duration, same recording
    uid = await _mk_user()
    await db.execute("INSERT INTO favorites (user_id, track_id) VALUES (:u, :t)", {"u": uid, "t": a})
    await db.execute("INSERT INTO favorites (user_id, track_id) VALUES (:u, :t)", {"u": uid, "t": b})

    res = await merge_duplicates()
    assert res["merged"] >= 1

    remaining = await db.fetch_all(
        "SELECT id FROM tracks WHERE title = 'Группа крови' AND 'Кино' = ANY(artists)"
    )
    assert len(remaining) == 1, "дубли должны слиться в один трек"
    canon = remaining[0]["id"]
    # favorite is preserved on the canonical track (no double, none lost)
    favs = await db.fetch_all("SELECT track_id FROM favorites WHERE user_id = :u", {"u": uid})
    assert [f["track_id"] for f in favs] == [canon]

    await db.execute("DELETE FROM favorites WHERE user_id = :u", {"u": uid})
    await db.execute("DELETE FROM users WHERE id = :u", {"u": uid})
    await db.execute("DELETE FROM tracks WHERE id = :c", {"c": canon})


@pytest.mark.asyncio
async def test_merges_homoglyph_artist_variants():
    """Latin/Cyrillic lookalikes must not split one recording (2341/2357)."""
    a = await _mk_track("Домик", ["Masha Hima"], 112)
    b = await _mk_track("Домик", ["Мashа Hima"], 112)
    uid = await _mk_user()
    await db.execute("INSERT INTO favorites (user_id, track_id) VALUES (:u, :t)", {"u": uid, "t": b})

    res = await merge_duplicates()
    assert res["merged"] >= 1

    remaining = await db.fetch_all(
        "SELECT id FROM tracks WHERE title = 'Домик' AND duration_sec = 112"
    )
    ids = {int(r["id"]) for r in remaining}
    assert len(ids & {a, b}) == 1
    fav = await db.fetch_one("SELECT track_id FROM favorites WHERE user_id = :u", {"u": uid})
    assert fav is not None and int(fav["track_id"]) in ids

    await db.execute("DELETE FROM favorites WHERE user_id = :u", {"u": uid})
    await db.execute("DELETE FROM users WHERE id = :u", {"u": uid})
    await db.execute("DELETE FROM tracks WHERE id = ANY(:ids)", {"ids": list(ids)})


@pytest.mark.asyncio
async def test_merges_artist_list_variants():
    """A collaboration packed into one artist string must share a key with
    separate artist entries (39338/39776 «Фейерверк»)."""
    a = await _mk_track("Фейерверк", ["Aikko Feat. Inspace,Катана Кассандры"], 302)
    b = await _mk_track("Фейерверк", ["aikko x Inspace x Катана Кассандры"], 302)
    uid = await _mk_user()
    await db.execute("INSERT INTO favorites (user_id, track_id) VALUES (:u, :t)", {"u": uid, "t": b})

    res = await merge_duplicates()
    assert res["merged"] >= 1

    remaining = await db.fetch_all(
        "SELECT id FROM tracks WHERE title = 'Фейерверк' AND duration_sec = 302"
    )
    # other tests may leave tracks named Фейерверк; pick the canonical id
    ids = {int(r["id"]) for r in remaining}
    assert a in ids or b in ids
    fav = await db.fetch_one("SELECT track_id FROM favorites WHERE user_id = :u", {"u": uid})
    assert fav is not None and int(fav["track_id"]) in ids
    assert len(ids & {a, b}) == 1

    await db.execute("DELETE FROM favorites WHERE user_id = :u", {"u": uid})
    await db.execute("DELETE FROM users WHERE id = :u", {"u": uid})
    await db.execute("DELETE FROM tracks WHERE id = ANY(:ids)", {"ids": list(ids)})


@pytest.mark.asyncio
async def test_does_not_move_corrupt_local_source_to_canonical():
    a = await _mk_track("Corrupt Merge", ["A"], 200)
    b = await _mk_track("Corrupt Merge", ["A"], 201)
    await db.execute(
        """
        INSERT INTO track_sources (track_id, connector, external_id, quality_tier, file_path, available)
        VALUES (:t, 'yandex', :e, 65, NULL, TRUE)
        """,
        {"t": a, "e": f"y-{uuid.uuid4().hex[:8]}"},
    )
    src = await db.execute_returning(
        """
        INSERT INTO track_sources (track_id, connector, external_id, quality_tier, file_path, available)
        VALUES (:t, 'local', :e, 100, '/tmp/corrupt.m4a', TRUE)
        RETURNING id
        """,
        {"t": b, "e": f"local::{uuid.uuid4().hex[:8]}"},
    )
    await db.execute(
        """
        INSERT INTO track_quality (track_id, local_source_id, file_path, tier, error)
        VALUES (:t, :s, '/tmp/corrupt.m4a', 0, 'truncated')
        """,
        {"t": b, "s": src[0]["id"]},
    )

    await merge_duplicates()

    local = await db.fetch_one(
        "SELECT 1 FROM track_sources WHERE track_id = :t AND connector = 'local'",
        {"t": a},
    )
    assert local is None, "corrupt local source must not move to canonical"
    assert await db.fetch_one("SELECT 1 FROM track_quality WHERE track_id = :t", {"t": b}) is None
    await db.execute("DELETE FROM tracks WHERE id = :i", {"i": a})


@pytest.mark.asyncio
async def test_does_not_merge_different_artists_or_versions():
    # different artists = covers → never merge
    x = await _mk_track("Heart Shaped Box", ["Nirvana"], 281)
    y = await _mk_track("Heart Shaped Box", ["Neovaii"], 290)
    # same artist but a clearly different edit (very different duration) → not merged
    z = await _mk_track("Группа крови", ["Кино"], 193)

    await merge_duplicates()

    ids = {r["id"] for r in await db.fetch_all("SELECT id FROM tracks WHERE title IN ('Heart Shaped Box','Группа крови')")}
    assert x in ids and y in ids, "каверы разных артистов нельзя сливать"
    assert z in ids, "сильно отличающуюся по длительности версию нельзя сливать"

    await db.execute("DELETE FROM tracks WHERE id = ANY(:i)", {"i": [x, y, z]})


async def _mk_artist(name: str, cover: str | None = None) -> int:
    rows = await db.execute_returning(
        "INSERT INTO artists (name, cover_url) VALUES (:n, :c) RETURNING id",
        {"n": name, "c": cover},
    )
    return rows[0]["id"]


@pytest.mark.asyncio
async def test_merges_duplicate_artists_preserves_references():
    from app.services.merge import merge_duplicate_artists

    a = await _mk_artist("Xoliday Boy")
    b = await _mk_artist("Хoliday Boy", cover="http://img/1.jpg")  # homoglyph duplicate
    uid = await _mk_user()
    await db.execute(
        "INSERT INTO favorite_artists (user_id, artist_id) VALUES (:u, :a)", {"u": uid, "a": b}
    )
    t = await _mk_track("Holiday Boy", ["Xoliday Boy"], 180)
    await db.execute(
        "INSERT INTO track_artists (track_id, artist_id) VALUES (:t, :a)", {"t": t, "a": a}
    )
    await db.execute(
        "INSERT INTO artist_sources (artist_id, connector, external_id) VALUES (:a, 'icm', 'x1')",
        {"a": a},
    )
    await db.execute(
        "INSERT INTO artist_sources (artist_id, connector, external_id) VALUES (:a, 'yandex', 'y1')",
        {"a": b},
    )

    res = await merge_duplicate_artists()
    assert res["merged"] >= 1

    remaining = await db.fetch_all("SELECT id FROM artists WHERE id IN (:a, :b)", {"a": a, "b": b})
    assert len(remaining) == 1, "гомоглифы должны слиться в одного артиста"
    canon = int(remaining[0]["id"])

    fav = await db.fetch_one(
        "SELECT artist_id FROM favorite_artists WHERE user_id = :u", {"u": uid}
    )
    assert fav is not None and int(fav["artist_id"]) == canon
    ta = await db.fetch_one(
        "SELECT artist_id FROM track_artists WHERE track_id = :t", {"t": t}
    )
    assert ta is not None and int(ta["artist_id"]) == canon
    srcs = await db.fetch_all(
        "SELECT connector FROM artist_sources WHERE artist_id = :i ORDER BY connector",
        {"i": canon},
    )
    assert sorted(s["connector"] for s in srcs) == ["icm", "yandex"], "источники не потеряны"

    # cleanup
    await db.execute("DELETE FROM favorite_artists WHERE user_id = :u", {"u": uid})
    await db.execute("DELETE FROM users WHERE id = :u", {"u": uid})
    await db.execute("DELETE FROM artist_sources WHERE artist_id = :i", {"i": canon})
    await db.execute("DELETE FROM track_artists WHERE track_id = :t", {"t": t})
    await db.execute("DELETE FROM tracks WHERE id = :t", {"t": t})
    await db.execute("DELETE FROM artists WHERE id = :i", {"i": canon})


@pytest.mark.asyncio
async def test_artist_merge_dry_run_touches_nothing():
    from app.services.merge import merge_duplicate_artists

    a = await _mk_artist("dryrun star")
    b = await _mk_artist("DRYRUN star")
    res = await merge_duplicate_artists(dry_run=True)
    assert res["merged"] >= 1
    left = await db.fetch_all("SELECT id FROM artists WHERE id IN (:a, :b)", {"a": a, "b": b})
    assert len(left) == 2
    await db.execute("DELETE FROM artists WHERE id = :a", {"a": a})
    await db.execute("DELETE FROM artists WHERE id = :b", {"b": b})


@pytest.mark.asyncio
async def test_backfill_album_types():
    from app.repair import backfill_album_types

    rows = await db.execute_returning(
        "INSERT INTO artists (name) VALUES ('typecheck') RETURNING id"
    )
    aid = rows[0]["id"]
    alb = await db.execute_returning(
        "INSERT INTO albums (title, artist_id, track_count) VALUES ('One Song', :a, 1) RETURNING id",
        {"a": aid},
    )
    res = await backfill_album_types()
    assert res["filled"] >= 1
    row = await db.fetch_one("SELECT album_type FROM albums WHERE id = :i", {"i": alb[0]["id"]})
    assert row["album_type"] == "single"
    await db.execute("DELETE FROM albums WHERE id = :i", {"i": alb[0]["id"]})
    await db.execute("DELETE FROM artists WHERE id = :i", {"i": aid})


@pytest.mark.asyncio
async def test_merges_album_title_variants_preserves_refs():
    """«X - Single» and «X» are one release — merge, keep refs (A3/B2)."""
    from app.services.merge import merge_duplicate_albums

    rows = await db.execute_returning(
        "INSERT INTO artists (name) VALUES ('albmerge') RETURNING id"
    )
    aid = rows[0]["id"]
    a = await db.execute_returning(
        "INSERT INTO albums (title, artist_id, year, cover_url) "
        "VALUES ('LOCA LA VIDA', :a, 2026, 'http://img/1.jpg') RETURNING id",
        {"a": aid},
    )
    b = await db.execute_returning(
        "INSERT INTO albums (title, artist_id, year) "
        "VALUES ('LOCA LA VIDA - Single', :a, 2026) RETURNING id",
        {"a": aid},
    )
    uid = await _mk_user()
    await db.execute(
        "INSERT INTO favorite_albums (user_id, album_id) VALUES (:u, :b)", {"u": uid, "b": b[0]["id"]}
    )
    await db.execute(
        "INSERT INTO album_sources (album_id, connector, external_id) VALUES (:b, 'icm', 'x9')",
        {"b": b[0]["id"]},
    )

    res = await merge_duplicate_albums()
    assert res["merged"] >= 1

    left = await db.fetch_all(
        "SELECT id FROM albums WHERE id IN (:a, :b)", {"a": a[0]["id"], "b": b[0]["id"]}
    )
    assert len(left) == 1, "варианты названия одного релиза должны слиться"
    canon = int(left[0]["id"])
    fav = await db.fetch_one(
        "SELECT album_id FROM favorite_albums WHERE user_id = :u", {"u": uid}
    )
    assert fav is not None and int(fav["album_id"]) == canon
    src = await db.fetch_one(
        "SELECT connector FROM album_sources WHERE album_id = :i", {"i": canon}
    )
    assert src is not None, "источник не потерян"

    await db.execute("DELETE FROM favorite_albums WHERE user_id = :u", {"u": uid})
    await db.execute("DELETE FROM users WHERE id = :u", {"u": uid})
    await db.execute("DELETE FROM album_sources WHERE album_id = :i", {"i": canon})
    await db.execute("DELETE FROM albums WHERE id = :i", {"i": canon})
    await db.execute("DELETE FROM artists WHERE id = :i", {"i": aid})


@pytest.mark.asyncio
async def test_album_merge_respects_year_split():
    """A 2019 single and a 2020 album with the same name stay separate."""
    from app.services.merge import merge_duplicate_albums

    rows = await db.execute_returning(
        "INSERT INTO artists (name) VALUES ('albmerge2') RETURNING id"
    )
    aid = rows[0]["id"]
    a = await db.execute_returning(
        "INSERT INTO albums (title, artist_id, year) VALUES ('Same Name', :a, 2019) RETURNING id",
        {"a": aid},
    )
    b = await db.execute_returning(
        "INSERT INTO albums (title, artist_id, year) VALUES ('Same Name', :a, 2020) RETURNING id",
        {"a": aid},
    )
    res = await merge_duplicate_albums()
    left = await db.fetch_all(
        "SELECT id FROM albums WHERE id IN (:a, :b)", {"a": a[0]["id"], "b": b[0]["id"]}
    )
    assert len(left) == 2, f"разные годы не должны сливаться (merged={res})"
    await db.execute("DELETE FROM albums WHERE artist_id = :a", {"a": aid})
    await db.execute("DELETE FROM artists WHERE id = :a", {"a": aid})
