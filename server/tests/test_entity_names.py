"""Artist lookups must not scan the whole artists table per upsert.

The search pipeline persists artists/albums for every query; the old
`_artist_by_name` loaded and normalized ~56k rows each time, which pushed
searches past the Android client's 20 s timeout. These tests pin the cached
index + fallback behaviour.
"""

from __future__ import annotations

import uuid

import pytest

from app import db
from app.services import dto, entity_upsert
from app.services.dedup import normalize_artist


@pytest.fixture(autouse=True)
async def _schema():
    await db.apply_schema()
    dto._idx_cache.clear()
    yield
    dto._idx_cache.clear()
    await db.close_engine()


async def _artist(name: str) -> int:
    rows = await db.execute_returning(
        "INSERT INTO artists (name) VALUES (:n) RETURNING id", {"n": name}
    )
    return int(rows[0]["id"])


async def test_artist_lookup_uses_cached_index_without_full_scan(monkeypatch):
    name = f"Index Artist {uuid.uuid4().hex[:8]}"
    artist_id = await _artist(name)
    index = await dto._artist_id_index()
    assert index[normalize_artist(name)] == artist_id

    original_fetch_all = db.fetch_all

    async def no_full_scan(sql, *args, **kwargs):
        if "LIMIT 100000" in sql:
            raise AssertionError("_artist_by_name must not scan all artists")
        return await original_fetch_all(sql, *args, **kwargs)

    monkeypatch.setattr(db, "fetch_all", no_full_scan)
    row = await entity_upsert._artist_by_name(name)
    assert row is not None
    assert int(row["id"]) == artist_id


async def test_artist_lookup_falls_back_and_remembers_new_rows():
    index = await dto._artist_id_index()
    name = f"Late Artist {uuid.uuid4().hex[:8]}"
    assert normalize_artist(name) not in index
    artist_id = await _artist(name)

    row = await entity_upsert._artist_by_name(name)
    assert row is not None
    assert int(row["id"]) == artist_id
    # The fallback must feed the cached map so the next search is O(1).
    assert dto._idx_cache["artists"][1][normalize_artist(name)] == artist_id


async def test_remember_album_updates_cached_index():
    artist_name = f"Album Artist {uuid.uuid4().hex[:8]}"
    artist_id = await _artist(artist_name)
    rows = await db.execute_returning(
        "INSERT INTO albums (title, artist_id) VALUES (:t, :a) RETURNING id",
        {"t": "Memory Album", "a": artist_id},
    )
    album_id = int(rows[0]["id"])
    await dto._album_id_index()
    dto.remember_album("Memory Album", artist_name, album_id)
    from app.services.dedup import normalize_title_or_raw

    key = (normalize_title_or_raw("Memory Album"), normalize_artist(artist_name))
    assert dto._idx_cache["albums"][1][key] == album_id
