"""Normalization rules (offline): titles, artist splitting, dedup keys."""

from __future__ import annotations

import os

os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://kwhale:kwhale_v3_pg@localhost:5433/kwhale")

from app.services.dedup import round_robin, track_identity_key  # noqa: E402
from app.services.normalize import (  # noqa: E402
    dedup_artist_key,
    dedup_title_key,
    split_artists,
    strip_version_junk,
)


def test_strip_feat_clauses():
    assert strip_version_junk("Song (feat. Someone)") == "Song"
    assert strip_version_junk("Song [ft. Other]") == "Song"
    assert strip_version_junk("Song (featuring A & B)") == "Song"


def test_strip_remaster_and_explicit():
    assert strip_version_junk("Song (2011 Remaster)") == "Song (2011 Remaster)"
    assert strip_version_junk("Song - Remastered 2011") == "Song - Remastered 2011"
    assert strip_version_junk("Song (Explicit)") == "Song"
    assert strip_version_junk("Song (Clean)") == "Song"
    assert strip_version_junk("Song (Explicit) (2011 Remaster)") == "Song (2011 Remaster)"


def test_keep_meaningful_versions():
    assert strip_version_junk("Song (Live at Wembley)") == "Song (Live at Wembley)"
    assert strip_version_junk("Song (Club Mix)") == "Song (Club Mix)"


def test_split_artists():
    assert split_artists("A feat. B") == ["A", "B"]
    assert split_artists("A & B, C") == ["A", "B", "C"]
    assert split_artists("A vs. B") == ["A", "B"]
    assert split_artists("Solo") == ["Solo"]
    assert split_artists("") == []


def test_dedup_keys_match_across_styles():
    assert dedup_title_key("One More Time! (Explicit)") == dedup_title_key("One More Time (Clean)")
    assert dedup_title_key("One More Time! (2021 Remaster)") != dedup_title_key("One More Time")
    assert dedup_artist_key("Daft Punk feat. Someone") == dedup_artist_key("Daft Punk")
    assert dedup_artist_key("Mumiy Troll") == dedup_artist_key("mumiy troll")


def test_track_identity_key_splits_multiartist():
    a = track_identity_key("Song", ["A, B"], 100)
    b = track_identity_key("Song", ["A & B"], 100)
    assert a[1] == b[1] == ("a", "b")
    # feat-clause in the title lands on the same identity as the clean title
    assert track_identity_key("Song (feat. B)", ["A"], 100)[0] == track_identity_key(
        "Song", ["A"], 100
    )[0]


def test_round_robin_fair_truncation():
    def meta(conn, ext, tier):
        from app.connectors.base import TrackMeta

        return TrackMeta(
            connector=conn, external_id=ext, title=f"t{ext}", artists=["x"], quality_tier=tier
        )

    groups = [
        {"winner": meta("icm", f"i{i}", 65), "sources": []} for i in range(4)
    ] + [{"winner": meta("yandex", f"y{i}", 65), "sources": []} for i in range(4)] + [
        {"winner": meta("zvuk", f"z{i}", 65), "sources": []} for i in range(4)
    ]
    picked = round_robin(groups, 6)
    winners = [g["winner"].connector for g in picked]
    assert winners.count("icm") == winners.count("yandex") == winners.count("zvuk") == 2
    # order inside a bucket is preserved
    ids = [g["winner"].external_id for g in picked]
    assert ids.index("i0") < ids.index("i1")


def test_derive_album_type_by_track_count():
    from app.services.normalize import derive_album_type

    assert derive_album_type("Song", 1) == "single"
    assert derive_album_type("EP Thing", 4) == "ep"
    assert derive_album_type("LP", 12) == "album"
    assert derive_album_type("Unknown") == "album"


def test_derive_album_type_explicit_wins():
    from app.services.normalize import derive_album_type

    assert derive_album_type("X", 12, "single") == "single"
    assert derive_album_type("X (Single)", 12) == "single"
    assert derive_album_type("X (EP)", 1) == "ep"
    assert derive_album_type("X", 1, "compilation") == "compilation"


def test_derive_album_type_title_marker():
    from app.services.normalize import derive_album_type

    assert derive_album_type("LOCA LA VIDA - Single", 12) == "single"
    assert derive_album_type("Something (EP)", 12) == "ep"
    assert derive_album_type("Album — EP", 12) == "ep"
    assert derive_album_type("Not A Single Thing", 12) == "album"
