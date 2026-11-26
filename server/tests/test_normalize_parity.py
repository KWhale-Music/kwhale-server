"""SQL/Python artist-normalization parity.

The artist-page track query (`services/artists.py::_db_artist_track_ids`)
cross-compares two normalizations of the same artist name: the Python key
(`services.dedup.normalize_artist` = `dedup_artist_key`, homoglyph folding +
NFKD) and the SQL generated column `tracks.artists_norm` built from
`artist_norm_imm()` (unaccent + й→и + strip). When they disagree the LIKE
prefilter matches nothing and the artist page shows zero tracks while albums
keep working — the «Тёмный Принц» bug (2026-11-18).

Stock PostgreSQL unaccent() folds Cyrillic «ё» to Cyrillic «е» and knows
nothing about the Latin look-alikes (т→t, н→h, …) that `dedup_artist_key`
folds, so SQL must apply the SAME homoglyph table first. Both sides of every
comparison must produce the same key for the same name; these tests pin that.
"""

from __future__ import annotations

import pytest

from app import db
from app.services.dedup import normalize_artist

PARITY_CORPUS = [
    # the reported bug: «ё» in the name
    "Тёмный Принц",
    "тёмный принц",
    "не тёмный принц",
    "kanji — тёмный принц",
    # Cyrillic with Latin look-alikes (homoglyphs)
    "Мэткид",
    "Слава КПСС",
    "Эмми",
    "Майкл",
    "Ёлки",
    # pure ASCII (always worked — regression guard)
    "madk1d",
    "sodu luv",
    "Kai Angel",
    # mixed scripts / real prod rows
    "Мashа",
    "Masha",
    'madk1d "так похуй"',
    "Движуха (VILLAIN, madk1d",
    "madk1d_and_molly..love",
    # punctuation, separators, diacritics
    "O'Brien",
    "AC/DC",
    "MONSTA X",
    "Tyler, The Creator",
    "Ünïcödé",
    "Måneskin",
    # degenerate
    "†",
    "",
]


def _py_key(name: str) -> str:
    # exactly what services/artists.py passes into the SQL LIKE pattern
    return normalize_artist(name).replace(" ", "")


@pytest.fixture(autouse=True)
async def _schema():
    await db.apply_schema()
    yield
    await db.close_engine()


@pytest.mark.asyncio
@pytest.mark.parametrize("name", PARITY_CORPUS)
async def test_artist_norm_matches_python_key(name: str):
    row = await db.fetch_one(
        "SELECT artist_norm(:n) AS norm, artist_norm_imm(:n) AS norm_imm", {"n": name}
    )
    assert row is not None
    assert row["norm"] == _py_key(name), f"artist_norm разошёлся с Python для {name!r}"
    assert row["norm_imm"] == _py_key(name), f"artist_norm_imm разошёлся с Python для {name!r}"


@pytest.mark.asyncio
async def test_homoglyph_variants_share_one_key():
    # the 2026-11-12 «homoglyph merge» promise: Мashа == Masha on BOTH sides
    assert _py_key("Мashа") == _py_key("Masha")
    row = await db.fetch_one(
        "SELECT artist_norm('Мashа') = artist_norm('Masha') AS eq,"
        " artist_norm_imm('Мashа') = artist_norm_imm('Masha') AS eq_imm"
    )
    assert row["eq"] is True
    assert row["eq_imm"] is True


@pytest.mark.asyncio
async def test_yo_and_plain_e_fold_identically():
    # «ё» must not become a separate letter from «е» (or «e») on either side
    row = await db.fetch_one(
        "SELECT artist_norm('Тёмный') = artist_norm('Темный') AS eq,"
        " artist_norm('Тёмный') = artist_norm('Temhый') AS eq_latin"
    )
    assert row["eq"] is True
    assert row["eq_latin"] is True
