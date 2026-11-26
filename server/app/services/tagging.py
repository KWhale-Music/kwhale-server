"""Auto-tagging: canonical metadata -> file tags, before the file is written.

DB (tracks row) is the source of truth. Tags are ALWAYS written (SoundCloud
files ship untagged; ICM files get normalised). Runs on the staged file in
_incoming/ so a crash never leaves a half-tagged file in the tree.

Formats: mp3 (ID3), flac (vorbis), m4a/mp4 (mp4 atoms). Cover is embedded
(APIC / METADATA_BLOCK_PICTURE / covr) from tracks.cover_url.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger("kwhale.tagging")

_UNSAFE_RE = re.compile(r'[\\/:*?"<>|\x00-\x1f]')
_SPACES_RE = re.compile(r"\s+")


def sanitize_segment(name: str, max_len: int = 120) -> str:
    """Sanitise a single path segment (artist/album/title)."""
    name = unicodedata.normalize("NFKD", name)
    name = _UNSAFE_RE.sub("_", name)
    name = _SPACES_RE.sub(" ", name).strip().strip("._ ")
    name = unicodedata.normalize("NFC", name)  # recompose after NFKD split
    if len(name) > max_len:
        name = name[:max_len].rstrip("._ ")
    return name


@dataclass
class Canonical:
    """Canonical metadata for one track (from tracks + track_sources rows)."""

    title: str
    artists: list[str] = field(default_factory=list)
    album: str | None = None
    year: int | None = None
    track_number: int | None = None
    track_total: int | None = None
    disc_number: int | None = None
    disc_total: int | None = None
    genre: str | None = None
    cover_url: str | None = None
    isrc: str | None = None
    explicit: bool | None = None
    lyrics: str | None = None  # plain text (USLT / ©lyr / LYRICS)
    lyrics_lrc: str | None = None  # synced LRC (mp3 SYLT / flac LYRICS)

    @property
    def album_artist(self) -> str:
        return self.artists[0] if self.artists else "Unknown Artist"

    def file_path(
        self, ext: str, existing: set[str] | None = None, suffix_override: str | None = None
    ) -> Path:
        """Managed-tree path for this track. existing = lowercase relative paths."""
        artist = sanitize_segment(self.album_artist, 80)
        album = sanitize_segment(self.album, 100) if self.album else "_Singles"
        if self.album and self.year:
            album = f"{sanitize_segment(self.album, 100)} ({self.year})"
        elif not self.album:
            album = "_Singles"

        title = sanitize_segment(self.title, 100)
        prefix = f"{self.track_number:02d} - " if self.track_number else ""
        stem = f"{prefix}{title}" or "Unknown Title"
        rel = f"{artist}/{album}/{stem}.{ext}"

        if suffix_override:
            rel = f"{artist}/{album}/{stem} {suffix_override}.{ext}"
        elif existing and rel.lower() in existing:
            rel = f"{artist}/{album}/{stem} [{self.track_id_hash}].{ext}"
        return Path(rel)

    track_id_hash: str = ""  # set by caller for collision suffix (short id)


def cover_mime_hint(url: str | None) -> tuple[str, str]:
    """(mime, ext) for the cover fetch."""
    if url and url.endswith(".png"):
        return "image/png", "png"
    return "image/jpeg", "jpg"


async def fetch_cover(url: str | None, proxy: str | None = None) -> bytes | None:
    """Download cover art bytes (10s timeout, best-effort, routed via httpcore)."""
    if not url:
        return None
    from .. import httpcore

    try:
        if proxy is not None:
            import httpx

            async with httpx.AsyncClient(timeout=10.0, proxy=proxy, follow_redirects=True) as c:
                r = await c.get(url)
                if r.status_code == 200 and r.content:
                    return r.content
        else:
            bucket = httpcore.connector_for_url(url)
            r = await httpcore.client(bucket).get(url, timeout=10.0)
            if r.status_code == 200 and r.content:
                return r.content
    except Exception as e:
        log.warning("cover fetch %s: %s", url, e)
    return None


def write_tags(path: Path, canonical: Canonical, cover: bytes | None = None) -> None:
    """Embed canonical tags + cover. ALWAYS overwrites (DB is authoritative)."""
    ext = path.suffix.lower()
    title = canonical.title or "Unknown Title"
    artist = ", ".join(canonical.artists) if canonical.artists else "Unknown Artist"
    albumartist = canonical.album_artist
    album = canonical.album or ""
    trck = f"{canonical.track_number}" if canonical.track_number else ""
    if canonical.track_number and canonical.track_total:
        trck = f"{canonical.track_number}/{canonical.track_total}"
    genre = canonical.genre or ""

    common = {
        "title": [title],
        "artist": [artist],
        "albumartist": [albumartist],
    }
    if album:
        common["album"] = [album]
    if canonical.year:
        common["date"] = [str(canonical.year)]
    if trck:
        common["tracknumber"] = [trck]
    if genre:
        common["genre"] = [genre]

    if ext == ".mp3":
        _tag_mp3(path, common, cover, canonical)
    elif ext == ".flac":
        _tag_flac(path, common, cover, canonical)
    elif ext in (".m4a", ".mp4"):
        _tag_mp4(path, common, cover, canonical)
    else:
        log.warning("unsupported container %s — tags skipped", ext)


def _tag_mp3(
    path: Path, tags: dict[str, list[str]], cover: bytes | None, canonical: Canonical
) -> None:
    import mutagen.easyid3
    import mutagen.id3

    # fresh downloads ship without any ID3 header — create one first
    try:
        mutagen.id3.ID3(str(path))
    except mutagen.id3.ID3NoHeaderError:
        mutagen.id3.ID3().save(str(path))

    mf = mutagen.easyid3.EasyID3(str(path))
    mf.update(tags)
    mf.save()

    audio = mutagen.id3.ID3(str(path))
    if cover:
        audio.add(mutagen.id3.APIC(encoding=3, mime="image/jpeg", type=3, data=cover))
    if canonical.isrc:
        audio.add(mutagen.id3.TSRC(encoding=3, text=[canonical.isrc]))
    if canonical.disc_number:
        disc = f"{canonical.disc_number}"
        if canonical.disc_total:
            disc = f"{disc}/{canonical.disc_total}"
        audio.add(mutagen.id3.TPOS(encoding=3, text=[disc]))
    if canonical.lyrics:
        audio.add(mutagen.id3.USLT(encoding=3, lang="und", desc="", text=canonical.lyrics))
    if canonical.lyrics_lrc:
        lines = parse_lrc_times(canonical.lyrics_lrc)
        if lines:
            audio.add(
                mutagen.id3.SYLT(
                    encoding=3,
                    lang="und",
                    format=2,  # milliseconds
                    type=1,  # lyrics
                    desc="",
                    text=[(text, ms) for ms, text in lines],
                )
            )
    audio.save(str(path))


def _tag_flac(
    path: Path, tags: dict[str, list[str]], cover: bytes | None, canonical: Canonical
) -> None:
    import mutagen.flac

    mf = mutagen.flac.FLAC(str(path))
    mf.update(tags)
    if canonical.isrc:
        mf["isrc"] = [canonical.isrc]
    if canonical.disc_number:
        mf["discnumber"] = [str(canonical.disc_number)]
    if canonical.disc_total:
        mf["disctotal"] = [str(canonical.disc_total)]
    if canonical.track_total:
        mf["totaltracks"] = [str(canonical.track_total)]
    if canonical.lyrics or canonical.lyrics_lrc:
        mf["lyrics"] = [canonical.lyrics_lrc or canonical.lyrics or ""]
    if canonical.explicit is not None:
        mf["rating"] = ["1" if canonical.explicit else "0"]
    if cover:
        from mutagen.flac import Picture

        pic = Picture()
        pic.type = 3  # front cover
        pic.mime = "image/jpeg"
        pic.data = cover
        mf.add_picture(picture=pic)
    mf.save()


def _tag_mp4(
    path: Path, tags: dict[str, list[str]], cover: bytes | None, canonical: Canonical
) -> None:
    import mutagen.mp4

    key_map = {
        "title": "\xa9nam",
        "artist": "\xa9ART",
        "albumartist": "aART",
        "album": "\xa9alb",
        "date": "\xa9day",
        "genre": "\xa9gen",
    }
    mf = mutagen.mp4.MP4(str(path))
    for easy_key, values in tags.items():
        if easy_key == "tracknumber":
            continue  # mp4 uses a tuple atom, handled below
        mp4_key = key_map.get(easy_key)
        if mp4_key:
            mf[mp4_key] = values
    if canonical.track_number:
        mf["trkn"] = [(canonical.track_number, canonical.track_total or 0)]
    if canonical.disc_number:
        mf["disk"] = [(canonical.disc_number, canonical.disc_total or 0)]
    if canonical.isrc:
        mf["----:com.apple.iTunes:ISRC"] = [canonical.isrc.encode("utf-8")]
    if canonical.lyrics:
        mf["\xa9lyr"] = [canonical.lyrics]
    if canonical.explicit is not None:
        mf["rtng"] = [1 if canonical.explicit else 0]
    mf.save()
    if cover:
        mf2 = mutagen.mp4.MP4(str(path))
        mf2["covr"] = [mutagen.mp4.MP4Cover(cover, imageformat=mutagen.mp4.MP4Cover.FORMAT_JPEG)]
        mf2.save()


def parse_lrc_times(lrc: str) -> list[tuple[int, str]]:
    """[(ms, text)] from an LRC body — shared with providers.parse_lrc semantics."""
    from .providers import parse_lrc

    return parse_lrc(lrc)
