"""Manual pipeline: fuzzy match against existing tracks (needs Postgres)."""

import pytest

from app import db
from app.services.manual import _match_existing


@pytest.fixture(autouse=True)
async def _schema():
    await db.apply_schema()
    yield
    await db.close_engine()


@pytest.mark.asyncio
async def test_match_existing_artist_variants():
    rows = await db.execute_returning(
        "INSERT INTO tracks (title, artists, album, duration_sec) "
        "VALUES ('Test Unique Song Xyz', '{Justice,JUSTICE}', '†', 235) RETURNING id"
    )
    tid = rows[0]["id"]
    # comma-separated artist on the manual side must still match
    assert await _match_existing("Test Unique Song Xyz", "Justice, JUSTICE", 235) == tid
    assert await _match_existing("Test Unique Song Xyz", "JUSTICE", None) == tid
    # wrong artist must not
    assert await _match_existing("Test Unique Song Xyz", "Daft Punk", 235) is None
    # wrong duration must not
    assert await _match_existing("Test Unique Song Xyz", "Justice", 400) is None
    await db.execute("DELETE FROM tracks WHERE id = :t", {"t": tid})


@pytest.mark.asyncio
async def test_fingerprint_duration_read(tmp_path):
    from app.services.fingerprint import _read_existing_tags

    # tiny valid mp3
    frame = b"\xff\xfb\x90\x00" + b"\x00" * 413
    data = b"ID3\x03\x00\x00\x00\x00\x00\x00" + frame * 20
    p = tmp_path / "t.mp3"
    p.write_bytes(data)
    meta = _read_existing_tags(p)
    assert "duration" in meta  # duration read from stream info (may be 0.0 here)
