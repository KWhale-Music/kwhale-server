"""Album covers: shared URLs across unrelated titles must be repairable.

The report: some albums on an artist card show another album's artwork
(«Shh...»). Album merges/backfills left 59 cover URLs shared by unrelated
titles; the repair only trusts evidence — the album's own tracks' covers.
"""

from __future__ import annotations

import pytest

from app import db
from app.repair import covers_audit


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


async def _mk_album(title: str, artist_id: int, cover: str) -> int:
    rows = await db.execute_returning(
        "INSERT INTO albums (title, artist_id, cover_url) VALUES (:t, :a, :c) RETURNING id",
        {"t": title, "a": artist_id, "c": cover},
    )
    return rows[0]["id"]


async def _mk_track(title: str, album: str, artists: list[str], cover: str) -> int:
    rows = await db.execute_returning(
        "INSERT INTO tracks (title, artists, album, duration_sec, cover_url, last_seen_at) "
        "VALUES (:t, CAST(:a AS text[]), :al, 200, :c, NOW()) RETURNING id",
        {"t": title, "a": artists, "al": album, "c": cover},
    )
    return rows[0]["id"]


@pytest.mark.asyncio
async def test_covers_audit_reports_shared_and_fixes_from_track_evidence():
    aid = await _mk_artist("Shh Artist")
    shared = "https://img.test/shared.jpg"
    shh = await _mk_album("Shh...", aid, shared)
    other = await _mk_album("Other Album", aid, shared)
    real = "https://img.test/shh-real.jpg"
    await _mk_track("track one", "Shh...", ["Shh Artist"], real)
    await _mk_track("track two", "Shh...", ["Shh Artist"], real)

    report = await covers_audit(fix=False, dry_run=True)
    flagged = {item["album_id"] for item in report["suspicious"]}
    assert shh in flagged and other in flagged

    fixed = await covers_audit(fix=True)
    assert any(entry["album_id"] == shh for entry in fixed["applied"])

    row = await db.fetch_one("SELECT cover_url FROM albums WHERE id = :i", {"i": shh})
    assert row["cover_url"] == real, "обложка должна восстановиться из треков альбома"
    row2 = await db.fetch_one("SELECT cover_url FROM albums WHERE id = :i", {"i": other})
    assert row2["cover_url"] == shared, "без доказательств альбом не трогаем"


@pytest.mark.asyncio
async def test_covers_audit_ignores_title_spelling_variants():
    aid = await _mk_artist("Variant Artist")
    shared = "https://img.test/one.jpg"
    first = await _mk_album("Album", aid, shared)
    second = await _mk_album("Album - Single", aid, shared)

    report = await covers_audit(fix=False, dry_run=True)
    flagged = {item["album_id"] for item in report["suspicious"]}
    assert not ({first, second} & flagged), "X и X - Single — один альбом, обложка общая законно"
