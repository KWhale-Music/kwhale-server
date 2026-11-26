"""Dedup tests: ISRC matching, fuzzy normalisation, duration tolerance."""

import pytest

from app.connectors.base import TrackMeta
from app.services.dedup import dedup_key, dedup_tracks, normalize_artist, normalize_title


def meta(
    connector: str,
    ext_id: str,
    title: str,
    artists: list[str],
    dur: int = 320,
    isrc: str | None = None,
    tier: int = 30,
) -> TrackMeta:
    return TrackMeta(
        connector=connector,
        external_id=ext_id,
        title=title,
        artists=artists,
        album="",
        duration_sec=dur,
        isrc=isrc,
        cover_url="",
        quality_tier=tier,
    )


def test_normalize_title_strips_feat_and_punctuation():
    assert normalize_title("One More Time (feat. Romanthony)") == normalize_title("One More Time")
    assert normalize_title("Track!") == normalize_title("Track")


def test_normalize_artist_lowercases():
    assert normalize_artist("Daft Punk") == normalize_artist("daft punk")


def test_dedup_key_shape():
    assert "|" in dedup_key(meta("icm", "1", "Hello", ["A"]))


def test_isrc_merges_across_connectors():
    groups = dedup_tracks(
        [
            meta("icm", "icm1", "One More Time", ["Daft Punk"], isrc="FRXTR0003200"),
            meta("yandex", "354357", "One More Time", ["Daft Punk"], isrc="FRXTR0003200"),
        ]
    )
    assert len(groups) == 1
    assert {m.connector for m in groups[0]["sources"]} == {"icm", "yandex"}


def test_fuzzy_merges_on_title_artist_duration():
    groups = dedup_tracks(
        [
            meta("icm", "a", "One More Time!", ["Daft Punk"], dur=320, tier=30),
            meta("soundcloud", "b", "One More Time", ["daft punk"], dur=321, tier=15),
        ]
    )
    assert len(groups) == 1


def test_duration_gap_splits():
    # same title/artist, wildly different duration -> separate tracks
    groups = dedup_tracks(
        [
            meta("icm", "a", "Intro", ["Artist"], dur=30),
            meta("yandex", "b", "Intro", ["Artist"], dur=300),
        ]
    )
    assert len(groups) == 2


def test_icm_wins_over_higher_quality_remote():
    groups = dedup_tracks(
        [
            meta("ytmusic", "yt1", "Song", ["A"], tier=10),
            meta("yandex", "ya1", "Song", ["A"], tier=35),
            meta("icm", "icm1", "Song", ["A"], tier=30),
        ]
    )
    assert len(groups) == 1
    assert groups[0]["winner"].connector == "icm"
    assert len(groups[0]["sources"]) == 3


def test_missing_duration_does_not_block_merge():
    groups = dedup_tracks(
        [
            meta("icm", "a", "Song", ["A"], dur=0),
            meta("yandex", "b", "Song", ["A"], dur=320),
        ]
    )
    assert len(groups) == 1


def test_different_isrc_different_tracks():
    groups = dedup_tracks(
        [
            meta("icm", "a", "Song", ["A"], isrc="AAA"),
            meta("yandex", "b", "Song", ["A"], dur=320, isrc="BBB"),
        ]
    )
    assert len(groups) == 2


@pytest.mark.asyncio
async def test_no_results_no_crash():
    groups = dedup_tracks([])
    assert groups == []


@pytest.mark.asyncio
async def test_progressive_search_first_wave_and_background_dedup(monkeypatch):
    import asyncio

    from app.services import search_service

    class FastConnector:
        async def is_available(self):
            return True

        async def search(self, query, limit):
            await asyncio.sleep(0.02)
            return [meta("yandex", "ya-1", "Jealous", ["Eyedress"], dur=122, isrc="US123")]

        async def search_albums(self, query, limit):
            return []

        async def search_artists(self, query, limit):
            return []

    class SlowConnector:
        async def is_available(self):
            return True

        async def search(self, query, limit):
            await asyncio.sleep(0.6)
            return [
                meta("icm", "icm-1", "Jealous", ["Eyedress"], dur=122, isrc="US123", tier=30),
                meta("icm", "icm-2", "Romantic Lover", ["Eyedress"], dur=145, isrc="US999", tier=30),
            ]

        async def search_albums(self, query, limit):
            return []

        async def search_artists(self, query, limit):
            return []

    cache_store = {}

    async def fake_cache_get(key, default=None):
        return cache_store.get(key, default)

    async def fake_cache_put(key, connector, value, ttl_hours=None):
        cache_store[key] = value

    async def fake_active_connectors():
        return {"yandex": FastConnector(), "icm": SlowConnector()}

    upserted_groups = []

    async def fake_persist(
        query, limit, user_id, track_metas, album_metas, artist_metas, errors,
        partial=False, with_dtos=True,
    ):
        groups = dedup_tracks(track_metas)
        upserted_groups.append((partial, groups))
        return {
            "query": query,
            "tracks": {"items": [{"id": idx + 1, "title": g["winner"].title, "sources": [s.connector for s in g["sources"]]} for idx, g in enumerate(groups)], "errors": []},
            "albums": {"items": [], "errors": []},
            "artists": {"items": [], "errors": []},
            "errors": errors,
            "partial": partial,
        }

    monkeypatch.setattr(search_service, "_cache_get", fake_cache_get)
    monkeypatch.setattr(search_service, "cache_put", fake_cache_put)
    monkeypatch.setattr(search_service, "_active_connectors", fake_active_connectors)
    monkeypatch.setattr(search_service, "_persist_and_format", fake_persist)

    wave1 = await search_service.search("jealous", 20, user_id=1, wait_all=False)
    assert wave1["partial"] is True
    assert len(wave1["tracks"]["items"]) == 1
    assert wave1["tracks"]["items"][0]["sources"] == ["yandex"]

    wave2 = await search_service.search("jealous", 20, user_id=1, wait_all=True)
    assert wave2["partial"] is False
    assert len(wave2["tracks"]["items"]) == 2
    jealous_item = next(t for t in wave2["tracks"]["items"] if t["title"] == "Jealous")
    assert set(jealous_item["sources"]) == {"yandex", "icm"}


def test_collapse_track_rows_collapses_noisy_durations():
    """Same recording with durations 240/247 must show once (artist page x5)."""
    from app.services.dedup import collapse_track_rows

    rows = [
        {"id": 1, "title": "Группа крови", "artists": ["Кино"], "duration_sec": 240},
        {"id": 2, "title": "Группа крови", "artists": ["Кино"], "duration_sec": 247},
        {"id": 3, "title": "Группа крови", "artists": ["Кино"], "duration_sec": 243},
        {"id": 4, "title": "Звезда", "artists": ["Кино"], "duration_sec": 240},
    ]
    out = collapse_track_rows(rows)
    assert [r["id"] for r in out] == [1, 4]


def test_collapse_track_rows_keeps_distinct_versions():
    from app.services.dedup import collapse_track_rows

    rows = [
        {"id": 1, "title": "Song", "artists": ["A"], "duration_sec": 200},
        {"id": 2, "title": "Song (Live)", "artists": ["A"], "duration_sec": 300},
        {"id": 3, "title": "Song", "artists": ["A"], "duration_sec": 260},  # different recording
    ]
    out = collapse_track_rows(rows)
    assert [r["id"] for r in out] == [1, 2, 3]


def test_collapse_track_rows_prefers_first_row():
    """Rows arrive local/favorite-first: the first copy of a cluster wins."""
    from app.services.dedup import collapse_track_rows

    rows = [
        {"id": 7, "title": "Song", "artists": ["A"], "duration_sec": 201},
        {"id": 8, "title": "Song", "artists": ["A"], "duration_sec": 200},
    ]
    out = collapse_track_rows(rows)
    assert [r["id"] for r in out] == [7]
