"""Provider helpers (offline): LRC parser, query optimizer, text cleanup."""

from __future__ import annotations

import os

os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://kwhale:kwhale_v3_pg@localhost:5433/kwhale")

from app.services.providers import (  # noqa: E402
    _clean_genius_text,
    optimize_search_query,
    parse_lrc,
)


def test_optimize_search_query():
    assert optimize_search_query("Song (feat. X) [Remastered]") == "song"
    assert optimize_search_query("Prod. by Someone - Track") == "someone - track"
    assert optimize_search_query("Keep  Spaces") == "keep spaces"


def test_parse_lrc_basic():
    lrc = "[00:01.50] first line\n[00:05.000] second\n[00:07.1] third"
    lines = parse_lrc(lrc)
    assert lines == [(1500, "first line"), (5000, "second"), (7100, "third")]


def test_parse_lrc_offset_and_multi_timecodes():
    lrc = "[offset:500]\n[00:01.00][00:10.00] chorus\n[00:02.00] verse"
    lines = parse_lrc(lrc)
    assert (1500, "chorus") in lines
    assert (10500, "chorus") in lines
    assert (2500, "verse") in lines


def test_parse_lrc_word_timings_stripped():
    lrc = "[00:01.00] <00:01.00> word <00:01.40> by <00:01.80> word"
    lines = parse_lrc(lrc)
    assert lines == [(1000, "word by word")]


def test_parse_lrc_ignores_garbage():
    assert parse_lrc("[ar: Artist]\n[ti: Title]\nnot a lyric line without timecode") == []
    assert parse_lrc("") == []


def test_genius_text_cleanup():
    raw = "  Verse 1  \nline one\n\n\n[Chorus]\nsing it\n"
    cleaned = _clean_genius_text(raw)
    assert cleaned.startswith("Verse 1\nline one")
    assert "\n[Chorus]" in cleaned
    assert "\n\n\n" not in cleaned
