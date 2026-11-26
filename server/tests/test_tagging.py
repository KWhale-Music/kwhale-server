"""Tagging unit tests: sanitize + tag round-trip on tiny fixtures (no DB)."""

from pathlib import Path

from app.services.tagging import Canonical, sanitize_segment, write_tags


def test_sanitize_strips_unsafe():
    assert sanitize_segment('AC/DC: "Back"?') == "AC_DC_ _Back"
    assert len(sanitize_segment("a" * 300)) == 120


def test_sanitize_keeps_unicode():
    assert sanitize_segment("† (2022)") == "† (2022)"
    assert sanitize_segment("Björk") == "Björk"


def _make_mp3(tmp_path: Path) -> Path:
    """Minimal valid mp3 (silence, MPEG1 Layer3 128k 44.1k, ~0.03s)."""
    # 32 frames of valid silence mp3
    frame = b"\xff\xfb\x90\x00" + b"\x00" * 413  # 417 bytes per frame @128k
    data = b"ID3\x03\x00\x00\x00\x00\x00\x00" + frame * 20
    p = tmp_path / "silence.mp3"
    p.write_bytes(data)
    return p


def test_mp3_tag_roundtrip(tmp_path):
    p = _make_mp3(tmp_path)
    c = Canonical(
        title="One More Time",
        artists=["Daft Punk"],
        album="Discovery",
        year=2001,
        track_number=4,
        genre="Electronic",
    )
    write_tags(p, c, cover=b"\xff\xd8\xff\xe0FAKE")
    import mutagen

    mf = mutagen.File(str(p), easy=True)
    assert (mf.get("title") or [None])[0] == "One More Time"
    assert (mf.get("artist") or [None])[0] == "Daft Punk"
    assert (mf.get("albumartist") or [None])[0] == "Daft Punk"
    assert (mf.get("album") or [None])[0] == "Discovery"
    assert (mf.get("date") or [None])[0] == "2001"
    assert (mf.get("tracknumber") or [None])[0] == "4"
    assert (mf.get("genre") or [None])[0] == "Electronic"
    # cover embedded
    import mutagen.id3

    audio = mutagen.id3.ID3(str(p))
    assert len(audio.getall("APIC")) == 1


def test_canonical_path_singles():
    c = Canonical(title="Loose Track", artists=["X"])
    p = c.file_path("mp3")
    assert "_Singles" in p.as_posix()
    assert p.name == "Loose Track.mp3"


def test_mp3_untagged_file_gets_tags(tmp_path):
    """Fresh downloads have no ID3 header at all — tagging must still work."""
    frame = b"\xff\xfb\x90\x00" + b"\x00" * 413
    p = tmp_path / "untagged.mp3"
    p.write_bytes(frame * 20)  # no ID3 header
    c = Canonical(title="Fresh", artists=["A"], isrc="X" * 12)
    write_tags(p, c, cover=None)
    import mutagen

    mf = mutagen.File(str(p), easy=True)
    assert (mf.get("title") or [None])[0] == "Fresh"


def test_mp3_extended_tags_roundtrip(tmp_path):
    p = _make_mp3(tmp_path)
    c = Canonical(
        title="One More Time",
        artists=["Daft Punk"],
        album="Discovery",
        year=2001,
        track_number=4,
        track_total=14,
        disc_number=1,
        disc_total=2,
        genre="Electronic",
        isrc = "GBDUW0000059",
        explicit=False,
        lyrics="Na na na na na...",
        lyrics_lrc="[00:01.00] Na na na na na...",
    )
    write_tags(p, c, cover=b"\xff\xd8\xff\xe0FAKE")
    import mutagen.id3

    audio = mutagen.id3.ID3(str(p))
    assert str(audio.getall("TSRC")[0]) == "GBDUW0000059"
    assert str(audio.getall("TPOS")[0]) == "1/2"
    assert str(audio.getall("USLT")[0]) == "Na na na na na..."
    sylt = audio.getall("SYLT")
    assert len(sylt) == 1
    assert sylt[0].text == [("Na na na na na...", 1000)]
    # tracknumber carries the total
    import mutagen

    mf = mutagen.File(str(p), easy=True)
    assert (mf.get("tracknumber") or [None])[0] == "4/14"


def test_canonical_path_with_year_and_track():
    c = Canonical(title="Genesis", artists=["Justice"], album="†", year=2022, track_number=1)
    p = c.file_path("mp3")
    assert p.as_posix() == "Justice/† (2022)/01 - Genesis.mp3"


def test_canonical_path_collision():
    c = Canonical(title="Song", artists=["A"])
    p = c.file_path("mp3", existing={"song/song.mp3"}, suffix_override="[deadbeef]")
    assert p.name == "Song [deadbeef].mp3"
    p2 = c.file_path("mp3", existing={"a/song [deadbeef].mp3"})
    assert "[deadbeef]" not in p2.name
