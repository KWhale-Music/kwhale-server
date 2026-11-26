"""Search relevance rules: match first, popularity as a tie breaker."""

from __future__ import annotations

from app.services.search_rank import (
    dedupe_albums,
    dedupe_artists,
    match_score,
    pick_top,
    sort_artists,
    sort_tracks,
    track_score,
)


def test_match_score_exact_prefix_and_fuzzy():
    assert match_score("nirvana", "Nirvana") == 1.0
    assert match_score("nirv", "Nirvana") > match_score("nirwna", "Nirvana")
    assert match_score("smells like teen spirit", "Smells Like Teen Spirit (Remastered)") > 0.8
    assert match_score("nirvana", "MiyaGi") < 0.5


def test_exact_artist_beats_track_named_like_the_artist():
    artists = [{"id": "1", "name": "Nirvana", "album_count": 12}]
    tracks = [
        {
            "id": "2",
            "title": "Nirvana",
            "artists": [{"id": "9", "name": "MiyaGi"}],
            "is_favorite": True,
        },
        {
            "id": "3",
            "title": "Smells Like Teen Spirit",
            "artists": [{"id": "1", "name": "Nirvana"}],
        },
    ]
    top = pick_top("Nirvana", tracks, [], artists, [])
    assert top is not None
    assert top["type"] == "artist"
    assert top["artist"]["name"] == "Nirvana"


def test_artist_match_lifts_their_tracks_above_title_only_match():
    tracks = [
        {"id": "1", "title": "Nirvana", "artists": [{"name": "MiyaGi"}]},
        {"id": "2", "title": "Come As You Are", "artists": [{"name": "Nirvana"}]},
    ]
    ordered = sort_tracks("Nirvana", tracks)
    assert ordered[0]["id"] == "2"
    assert track_score("Nirvana", tracks[1]) > track_score("Nirvana", tracks[0])


def test_artist_sort_prefers_exact_name_over_popularity():
    artists = [
        {"id": "1", "name": "Nirvana Tribute Band", "album_count": 50},
        {"id": "2", "name": "Nirvana", "album_count": 1},
    ]
    ordered = sort_artists("Nirvana", artists)
    assert ordered[0]["id"] == "2"


def test_album_title_query_returns_album_not_artist():
    albums = [
        {
            "id": "10",
            "title": "Nevermind",
            "artists": [{"name": "Nirvana"}],
            "track_count": 13,
        }
    ]
    artists = [{"id": "1", "name": "Nirvana", "album_count": 12}]
    top = pick_top("Nevermind", [], albums, artists, [])
    assert top is not None
    assert top["type"] == "album"


def test_local_popularity_breaks_exact_artist_album_tie():
    artists = [{"id": "1", "name": "Nevermind", "album_count": 0}]
    albums = [
        {
            "id": "2",
            "title": "Nevermind",
            "artists": [{"name": "Nirvana"}],
            "track_count": 13,
        }
    ]
    popular_album = pick_top(
        "Nevermind", [], albums, artists, [], {"artist:1": 0.0, "album:2": 1.0}
    )
    assert popular_album is not None and popular_album["type"] == "album"

    popular_artist = pick_top(
        "Nevermind", [], albums, artists, [], {"artist:1": 1.0, "album:2": 0.0}
    )
    assert popular_artist is not None and popular_artist["type"] == "artist"


def test_newest_release_wins_on_equal_title_match():
    """«Holiday Boy»: the latest single must outrank the old track (B1)."""
    tracks = [
        {"id": "1", "title": "Holiday Boy", "artists": [{"name": "Xoliday Boy"}], "year": 2019},
        {"id": "2", "title": "Holiday Boy", "artists": [{"name": "Xoliday Boy"}], "year": 2024},
    ]
    ordered = sort_tracks("Holiday Boy", tracks)
    assert ordered[0]["id"] == "2"


def test_exact_match_still_outranks_newer_fuzzy():
    tracks = [
        {"id": "1", "title": "Holiday Boy", "artists": [{"name": "Xoliday Boy"}], "year": 2019},
        {"id": "2", "title": "Holiday Boy Remix Mashup Extended", "artists": [{"name": "DJ"}], "year": 2025},
    ]
    ordered = sort_tracks("Holiday Boy", tracks)
    assert ordered[0]["id"] == "1"


def test_pick_top_prefers_newer_on_tie():
    tracks = [
        {"id": "1", "title": "Holiday Boy", "artists": [{"name": "Xoliday Boy"}], "year": 2019},
        {"id": "2", "title": "Holiday Boy", "artists": [{"name": "Xoliday Boy"}], "year": 2024},
    ]
    top = pick_top("Holiday Boy", tracks, [], [], [])
    assert top is not None and top["type"] == "track"
    assert top["track"]["id"] == "2"


def test_dedupe_artists_collapses_same_name():
    artists = [
        {"id": "33709", "name": "Holidayboy", "album_count": 0},
        {"id": "33710", "name": "Holidayboy", "album_count": 3},
        {"id": "33711", "name": "Holidayboy7", "album_count": 0},
    ]
    out = dedupe_artists(artists)
    assert [a["id"] for a in out] == ["33710", "33711"]


def test_dedupe_artists_collapses_same_id_and_homoglyphs():
    artists = [
        {"id": "1", "name": "Xoliday Boy"},
        {"id": "1", "name": "Xoliday Boy"},  # same row twice in one response
        {"id": "2", "name": "Хoliday Boy"},  # Cyrillic homoglyph Х
    ]
    out = dedupe_artists(artists)
    assert [a["id"] for a in out] == ["1"]


def test_dedupe_artists_prefers_entry_with_cover():
    artists = [
        {"id": "1", "name": "Xoliday Boy", "avatar": {"id": "placeholder"}},
        {"id": "2", "name": "Xoliday Boy", "avatar": {"id": "abc123"}},
    ]
    out = dedupe_artists(artists)
    assert [a["id"] for a in out] == ["2"]


def test_dedupe_albums_collapses_same_title_artist():
    albums = [
        {"id": "1", "title": "Holiday Boy", "artists": [{"name": "Xoliday Boy"}], "track_count": 1, "cover": {"id": "placeholder"}},
        {"id": "2", "title": "Holiday Boy", "artists": [{"name": "Xoliday Boy"}], "track_count": 1, "cover": {"id": "abc"}},
        {"id": "3", "title": "Summer", "artists": [{"name": "Xoliday Boy"}], "track_count": 10},
    ]
    out = dedupe_albums(albums)
    assert [a["id"] for a in out] == ["2", "3"]
