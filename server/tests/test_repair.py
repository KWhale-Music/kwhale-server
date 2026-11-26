"""Library repair: source-less favorites get moved/re-searched, unavailable
sources are retried once, and dry-run changes nothing."""

from __future__ import annotations

import uuid
from types import SimpleNamespace

import pytest

from app import db
from app.connectors.base import TrackMeta
from app.services import repair


@pytest.fixture(autouse=True)
async def _schema():
    await db.apply_schema()
    await db.execute("TRUNCATE tracks CASCADE")
    await db.execute("DELETE FROM events")
    yield
    await db.close_engine()


async def _user() -> int:
    s = uuid.uuid4().hex[:8]
    rows = await db.execute_returning(
        "INSERT INTO users (username, password_hash, email, role) VALUES (:u, 'x', :e, 'user') RETURNING id",
        {"u": f"r-{s}", "e": f"r-{s}@example.com"},
    )
    return int(rows[0]["id"])


async def _track(title: str, artists: list[str], duration: int) -> int:
    rows = await db.execute_returning(
        "INSERT INTO tracks (title, artists, duration_sec) VALUES (:t, :a, :d) RETURNING id",
        {"t": title, "a": artists, "d": duration},
    )
    return int(rows[0]["id"])


async def _favorite(user_id: int, track_id: int) -> None:
    await db.execute(
        "INSERT INTO favorites (user_id, track_id) VALUES (:u, :t)",
        {"u": user_id, "t": track_id},
    )


async def _source(track_id: int, connector: str, ext: str, available: bool = True) -> None:
    await db.execute(
        """
        INSERT INTO track_sources (track_id, connector, external_id, available)
        VALUES (:t, :c, :e, :a)
        """,
        {"t": track_id, "c": connector, "e": ext, "a": available},
    )


async def test_moves_favorite_to_sibling_with_source():
    user = await _user()
    orphan = await _track("Repair Song", ["Repair Artist"], 200)
    sibling = await _track("Repair Song", ["Repair Artist"], 300)
    await _source(sibling, "yandex", f"y-{uuid.uuid4().hex[:8]}")
    await _favorite(user, orphan)

    result = await repair.repair_favorites()

    assert result["moved"] == 1
    row = await db.fetch_one(
        "SELECT track_id FROM favorites WHERE user_id = :u", {"u": user}
    )
    assert row is not None and int(row["track_id"]) == sibling
    gone = await db.fetch_one("SELECT 1 FROM tracks WHERE id = :i", {"i": orphan})
    assert gone is None


async def test_retries_unavailable_source_once(monkeypatch):
    user = await _user()
    track = await _track("Retry Song", ["Retry Artist"], 180)
    await _source(track, "icm", f"icm-{uuid.uuid4().hex[:8]}", available=False)
    await _favorite(user, track)

    class FakeConnector:
        async def resolve_stream(self, external_id):
            return SimpleNamespace(url="https://example.test/a.mp3")

    monkeypatch.setattr(repair, "get_connector", lambda name: FakeConnector())

    result = await repair.repair_favorites()

    assert result["reactivated"] == 1
    row = await db.fetch_one(
        "SELECT available FROM track_sources WHERE track_id = :t", {"t": track}
    )
    assert row is not None and row["available"] is True


async def test_conflicting_search_source_moves_to_canonical(monkeypatch):
    user = await _user()
    orphan = await _track("Conflict Song", ["Conflict Artist"], 200)
    canonical = await _track("Conflict Song", ["Conflict Artist"], 300)
    ext = f"c-{uuid.uuid4().hex[:8]}"
    await _source(canonical, "yandex", ext, available=False)
    await _favorite(user, orphan)

    meta = TrackMeta(
        connector="yandex",
        external_id=ext,
        title="Conflict Song",
        artists=["Conflict Artist"],
        duration_sec=200,
        quality_tier=65,
    )

    async def fake_fan_out(query, limit):
        return [meta]

    monkeypatch.setattr(repair.search_service, "_fan_out", fake_fan_out)

    result = await repair.repair_favorites()

    assert result["attached"] == 1
    row = await db.fetch_one(
        "SELECT track_id FROM favorites WHERE user_id = :u", {"u": user}
    )
    assert row is not None and int(row["track_id"]) == canonical
    source = await db.fetch_one(
        "SELECT available FROM track_sources WHERE connector = 'yandex' AND external_id = :e",
        {"e": ext},
    )
    assert source is not None and source["available"] is True
    assert await db.fetch_one("SELECT 1 FROM tracks WHERE id = :i", {"i": orphan}) is None


async def test_searches_and_attaches_source(monkeypatch):
    user = await _user()
    track = await _track("Attach Song", ["Attach Artist"], 190)
    await _favorite(user, track)

    meta = TrackMeta(
        connector="yandex",
        external_id=f"a-{uuid.uuid4().hex[:8]}",
        title="Attach Song",
        artists=["Attach Artist"],
        duration_sec=190,
        quality_tier=65,
    )

    async def fake_fan_out(query, limit):
        return [meta]

    monkeypatch.setattr(repair.search_service, "_fan_out", fake_fan_out)

    result = await repair.repair_favorites()

    assert result["attached"] == 1
    row = await db.fetch_one(
        "SELECT connector, external_id, available FROM track_sources WHERE track_id = :t",
        {"t": track},
    )
    assert row is not None
    assert row["connector"] == "yandex"
    assert row["available"] is True


async def test_dry_run_does_not_move_favorite():
    user = await _user()
    orphan = await _track("Dry Song", ["Dry Artist"], 200)
    sibling = await _track("Dry Song", ["Dry Artist"], 201)
    await _source(sibling, "yandex", f"d-{uuid.uuid4().hex[:8]}")
    await _favorite(user, orphan)

    result = await repair.repair_favorites(dry_run=True)

    assert result["moved"] == 1
    row = await db.fetch_one(
        "SELECT track_id FROM favorites WHERE user_id = :u", {"u": user}
    )
    assert row is not None and int(row["track_id"]) == orphan
    assert await db.fetch_one("SELECT 1 FROM tracks WHERE id = :i", {"i": sibling}) is not None
