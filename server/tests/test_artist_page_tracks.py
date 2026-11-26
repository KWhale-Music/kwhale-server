"""Artist page track lists must not silently return empty.

Regression for the «Тёмный Принц» bug (2026-11-18): the artist page showed
albums and singles but no tracks. `_db_artist_track_ids()` builds its LIKE
prefilter from the *Python* normalization key while the generated column
`tracks.artists_norm` is built from the *SQL* `artist_norm_imm()` — for
Cyrillic names («ё», homoglyph letters) the two disagreed and the prefilter
matched zero rows. ASCII names (madk1d, sodu luv) kept working, which is why
the bug looked selective.
"""

from __future__ import annotations

import pytest

from app import db
from app.services import artists as artists_service


@pytest.fixture(autouse=True)
async def _schema():
    await db.apply_schema()
    artists_service.clear_detail_cache()
    yield
    artists_service.clear_detail_cache()
    await db.close_engine()


@pytest.fixture(autouse=True)
def _no_sync(monkeypatch):
    # artist pages kick a background discography sync; keep tests offline
    async def _noop(*_args, **_kwargs):
        return None

    monkeypatch.setattr(artists_service, "schedule_artist_sync_if_due", _noop)


async def _mk_artist(name: str) -> int:
    rows = await db.execute_returning(
        "INSERT INTO artists (name) VALUES (:n) RETURNING id", {"n": name}
    )
    return rows[0]["id"]


async def _mk_track(title: str, artists: list[str], dur: int = 200) -> int:
    rows = await db.execute_returning(
        "INSERT INTO tracks (title, artists, album, duration_sec, last_seen_at) "
        "VALUES (:t, :a, '', :d, NOW()) RETURNING id",
        {"t": title, "a": artists, "d": dur},
    )
    return rows[0]["id"]


@pytest.mark.asyncio
async def test_cyrillic_yo_artist_gets_its_tracks():
    """The exact reported case: «тёмный принц» had 0 tracks on its page."""
    aid = await _mk_artist("тёмный принц")
    t1 = await _mk_track("трек один", ["тёмный принц"])
    t2 = await _mk_track("трек два", ["Тёмный Принц"])

    ids = await artists_service._db_artist_track_ids(aid, 5000)
    assert set(ids) >= {t1, t2}, "кириллический артист с «ё» должен получать свои треки"

    page = await artists_service.list_artist_tracks(aid, 50, 0, user_id=None)
    assert page is not None
    assert {int(item["id"]) for item in page["items"]} >= {t1, t2}


@pytest.mark.asyncio
async def test_ascii_artist_still_works():
    """madk1d / sodu luv worked before — keep them working."""
    aid = await _mk_artist("madk1d")
    tid = await _mk_track("song", ["madk1d"])
    ids = await artists_service._db_artist_track_ids(aid, 5000)
    assert tid in ids


@pytest.mark.asyncio
async def test_homoglyph_variant_track_is_found():
    """Track tagged «Мashа» must show on the page of the artist «Masha»."""
    aid = await _mk_artist("Masha")
    tid = await _mk_track("song", ["Мashа"])
    ids = await artists_service._db_artist_track_ids(aid, 5000)
    assert tid in ids


@pytest.mark.asyncio
async def test_composite_track_visible_from_member_page():
    """A row tagged «A & B» stays visible from the page of «A» (token semantics)."""
    aid = await _mk_artist("A")
    tid = await _mk_track("duet", ["A & B"])
    ids = await artists_service._db_artist_track_ids(aid, 5000)
    assert tid in ids


@pytest.mark.asyncio
async def test_lazy_tracks_survive_canonical_requery(monkeypatch):
    """Lazy-enriched ids must not be wiped when the DB re-query comes back short."""
    # unique name: the DB must have zero matching rows so the lazy path runs
    aid = await _mk_artist("lazy-принц-zzq")
    ghost = await _mk_track("провайдерный трек", ["ghost artist"])

    async def _fake_lazy(_aid: int, _name: str, _limit: int = 100) -> list[int]:
        return [ghost]

    async def _fake_albums(_aid: int, _name: str) -> list[int]:
        return []

    async def _has_sources(_aid: int) -> bool:
        return True

    async def _fake_latest(_aid: int, _name: str):
        return None

    async def _passthrough(coro, default, **_kw):
        return await coro

    monkeypatch.setattr(artists_service, "_lazy_top_tracks", _fake_lazy)
    monkeypatch.setattr(artists_service, "_lazy_artist_albums", _fake_albums)
    monkeypatch.setattr(artists_service, "_has_artist_sources", _has_sources)
    monkeypatch.setattr(artists_service, "_latest_release", _fake_latest)
    monkeypatch.setattr(artists_service, "_best_effort", _passthrough)

    detail = await artists_service.get_artist(aid, user_id=None)
    assert detail is not None
    top_ids = {int(item["id"]) for item in detail["top_tracks"]["items"]}
    assert ghost in top_ids, "лениво подтянутые треки не должны затираться повторным запросом"
