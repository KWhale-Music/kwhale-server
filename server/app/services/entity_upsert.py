"""Entity upserts: artists and albums (stable ids, sources[], last_seen TTL).

Fuzzy matching mirrors tracks: normalized name for artists,
normalized(title)+artist_id+year for albums. Repeated searches refresh
last_seen_at and merge sources; nothing is duplicated.
"""

from __future__ import annotations

import logging
import re
from datetime import date
from typing import Any

from .. import db
from ..connectors.base import AlbumMeta, ArtistMeta
from .dedup import normalize_artist
from .normalize import (
    album_title_key,
    derive_album_type,
    flatten_artist_names,
    split_artists_strict,
)

log = logging.getLogger("kwhale.entity")


async def upsert_artist(meta: ArtistMeta) -> int:
    """Insert/merge an artist by normalized name; refresh source row."""
    existing = await _artist_by_name(meta.name)
    if existing is None:
        rows = await db.execute_returning(
            """
            INSERT INTO artists (name, cover_url, genre, last_seen_at)
            VALUES (:n, :c, :g, NOW())
            RETURNING id
            """,
            {"n": meta.name, "c": meta.cover_url or None, "g": meta.genre},
        )
        artist_id = rows[0]["id"]
    else:
        artist_id = existing["id"]
        await db.execute(
            """
            UPDATE artists SET
                name = COALESCE(NULLIF(:n, ''), name),
                cover_url = COALESCE(:c, cover_url),
                genre = COALESCE(:g, genre),
                last_seen_at = NOW()
            WHERE id = :i
            """,
            {"n": meta.name, "c": meta.cover_url or None, "g": meta.genre, "i": artist_id},
        )
    from .dto import remember_artist

    remember_artist(meta.name, artist_id)
    await _upsert_artist_source(artist_id, meta.connector, meta.external_id)
    return artist_id


async def _artist_by_name(name: str) -> dict[str, Any] | None:
    """Find an artist by normalized name.

    The old implementation loaded and normalized every artist row on every
    lookup. A search persists several artists/albums, which made 20-30 full
    scans of a 56k-row table (~0.2 s each) and pushed search past the client's
    20 s timeout. Reuse the cached dto index; the SQL fallback only runs for a
    row created after the snapshot.
    """
    norm = normalize_artist(name)
    if not norm:
        return None
    from .dto import _artist_id_index, remember_artist

    index = await _artist_id_index()
    artist_id = index.get(norm)
    if artist_id is not None:
        row = await db.fetch_one(
            "SELECT id, name FROM artists WHERE id = :i", {"i": artist_id}
        )
        if row is not None:
            return row
        index.pop(norm, None)

    # Compare through the SQL normaliser on both sides: Python keeps spaces
    # while artist_norm() strips them, so passing the Python key directly
    # would miss every multi-word artist.
    row = await db.fetch_one(
        "SELECT id, name FROM artists "
        "WHERE artist_norm_imm(name) = artist_norm_imm(:n) LIMIT 1",
        {"n": name},
    )
    if row is not None:
        remember_artist(row["name"], row["id"])
    return row


async def _upsert_artist_source(artist_id: int, connector: str, external_id: str) -> None:
    await db.execute(
        """
        INSERT INTO artist_sources (artist_id, connector, external_id, updated_at)
        VALUES (:a, :c, :e, NOW())
        ON CONFLICT (connector, external_id) DO UPDATE
        SET artist_id = EXCLUDED.artist_id,
            available = TRUE,
            updated_at = NOW()
        """,
        {"a": artist_id, "c": connector, "e": external_id},
    )


async def upsert_album(meta: AlbumMeta, artist_id: int | None = None) -> int:
    """Insert/merge an album by normalized(title)+artist+year; refresh source."""
    resolved_artist_name = meta.artist_name or ""
    if artist_id is None and meta.artist_name:
        resolved = await resolve_artist_names([meta.artist_name])
        resolved_artist_name = resolved[0] if resolved else meta.artist_name
        artist_id = await upsert_artist(
            ArtistMeta(
                connector=meta.connector,
                external_id=f"album:{meta.external_id}",
                name=resolved_artist_name,
            )
        )
    if artist_id is None:
        artist_id = await _unknown_artist_id()

    release_date = None
    if meta.release_date:
        try:
            release_date = date.fromisoformat(str(meta.release_date)[:10])
        except ValueError:
            release_date = None
    album_type = (meta.album_type or "").strip().lower() or derive_album_type(
        meta.title, meta.track_count
    )
    existing = await _album_fuzzy(meta.title, artist_id, meta.year)
    if existing is None:
        rows = await db.execute_returning(
            """
            INSERT INTO albums (title, artist_id, year, release_date, cover_url, track_count, album_type, last_seen_at)
            VALUES (:t, :a, :y, :rd, :c, :tc, :at, NOW())
            RETURNING id
            """,
            {
                "t": meta.title,
                "a": artist_id,
                "y": meta.year,
                "rd": release_date,
                "c": meta.cover_url or None,
                "tc": meta.track_count,
                "at": album_type,
            },
        )
        album_id = rows[0]["id"]
    else:
        album_id = existing["id"]
        await db.execute(
            """
            UPDATE albums SET
                title = COALESCE(NULLIF(:t, ''), title),
                year = COALESCE(:y, year),
                release_date = COALESCE(:rd, release_date),
                cover_url = COALESCE(:c, cover_url),
                track_count = COALESCE(:tc, track_count),
                album_type = COALESCE(NULLIF(:at, ''), album_type),
                last_seen_at = NOW()
            WHERE id = :i
            """,
            {
                "t": meta.title,
                "y": meta.year,
                "rd": release_date,
                "c": meta.cover_url or None,
                "tc": meta.track_count,
                "at": album_type,
                "i": album_id,
            },
        )
    artist_name = meta.artist_name or ""
    if not artist_name and artist_id:
        artist_row = await db.fetch_one(
            "SELECT name FROM artists WHERE id = :i", {"i": artist_id}
        )
        artist_name = (artist_row or {}).get("name") or ""
    from .dto import remember_album

    remember_album(meta.title, artist_name, album_id)
    await _upsert_album_source(album_id, meta)
    return album_id


async def _album_fuzzy(
    title: str, artist_id: int | None, year: int | None
) -> dict[str, Any] | None:
    norm = album_title_key(title)
    if not norm or not artist_id:
        return None
    rows = await db.fetch_all(
        "SELECT id, title, year FROM albums WHERE artist_id = :a LIMIT 10000",
        {"a": artist_id},
    )
    for r in rows:
        if album_title_key(r["title"]) != norm:
            continue
        if year and r["year"] and abs(int(year) - int(r["year"])) > 1:
            continue
        if year and r["year"] and year != r["year"]:
            # Only fold ±1 year during search/ingestion when release doesn't look
            # like a single vs full album mismatch, preventing 2019 single + 2020 LP collisions.
            src_diff = await db.fetch_one(
                "SELECT 1 FROM album_sources WHERE album_id = :id LIMIT 1",
                {"id": r["id"]},
            )
            if not src_diff:
                continue
        return r
    return None


async def _upsert_album_source(album_id: int, meta: AlbumMeta) -> None:
    await db.execute(
        """
        INSERT INTO album_sources (album_id, connector, external_id, updated_at)
        VALUES (:a, :c, :e, NOW())
        ON CONFLICT (connector, external_id) DO UPDATE
        SET album_id = EXCLUDED.album_id,
            available = TRUE,
            updated_at = NOW()
        """,
        {"a": album_id, "c": meta.connector, "e": meta.external_id},
    )


async def _unknown_artist_id() -> int:
    row = await db.fetch_one("SELECT id FROM artists WHERE lower(name) = 'unknown artist' LIMIT 1")
    if row:
        return row["id"]
    rows = await db.execute_returning(
        "INSERT INTO artists (name) VALUES ('Unknown Artist') RETURNING id"
    )
    return rows[0]["id"]


# ── Source-less ensures (materialise rows for liked tracks / navigation) ─────


async def ensure_artist_by_name(name: str, cover_url: str | None = None) -> int | None:
    """Create/find an artist row by normalized name (no connector source)."""
    if not name.strip():
        return None
    existing = await _artist_by_name(name)
    if existing is not None:
        return existing["id"]
    rows = await db.execute_returning(
        "INSERT INTO artists (name, cover_url) VALUES (:n, :c) RETURNING id",
        {"n": name.strip(), "c": cover_url or None},
    )
    artist_id = rows[0]["id"]
    from .dto import remember_artist

    remember_artist(name, artist_id)
    return artist_id


async def ensure_album_by_title(
    title: str,
    artist_id: int | None,
    year: int | None = None,
    cover_url: str | None = None,
) -> int | None:
    """Create/find an album row by normalized title + artist (no source).

    The year is deliberately ignored when matching: connector/track years often
    disagree with the stored album, and creating a year-variant duplicate per
    like/backfill is worse than attaching to the existing row.
    """
    if not title.strip() or artist_id is None:
        return None
    existing = await _album_fuzzy(title, artist_id, None)
    if existing is not None:
        album_id = existing["id"]
    else:
        rows = await db.execute_returning(
            """
            INSERT INTO albums (title, artist_id, year, cover_url, album_type)
            VALUES (:t, :a, :y, :c, :at)
            RETURNING id
            """,
            {
                "t": title.strip(),
                "a": artist_id,
                "y": year,
                "c": cover_url or None,
                "at": derive_album_type(title),
            },
        )
        album_id = rows[0]["id"]
    artist_row = await db.fetch_one(
        "SELECT name FROM artists WHERE id = :i", {"i": artist_id}
    )
    if artist_row is not None:
        from .dto import remember_album

        remember_album(title, artist_row["name"], album_id)
    return album_id


_HAS_FEAT_RE = re.compile(r"\b(?:feat|ft|featuring)\b", re.IGNORECASE)


async def resolve_artist_names(
    raw_names: list[str],
    corroborating: list[list[str]] | None = None,
) -> list[str]:
    """Turn collab-packed names into participants — only with evidence.

    Sources pack collabs inconsistently: one sends ["Тимати,MOT,ЕГОР КРИД"],
    another ["Тимати","MOT","ЕГОР КРИД"]. But some real names contain
    separators ("Tyler, The Creator", "Simon & Garfunkel"), so blind splitting
    materialises junk artists. Rules:
      - "feat./ft./featuring" ALWAYS splits — the marker is unambiguous;
      - comma/&/vs split only when supported: another source's artist list
        carries ≥2 of the parts separately, or ≥2 parts already exist as
        artist rows. Otherwise the name stays whole — a wrong whole beats a
        shredded band.
    """
    names: list[str] = []
    for name in raw_names:
        parts = split_artists_strict(name)
        if len(parts) <= 1:
            if name and name not in names:
                names.append(name)
            continue
        if _HAS_FEAT_RE.search(name or ""):
            corroborated = True
        else:
            part_keys = {normalize_artist(p) for p in parts}
            part_keys.discard("")
            corroborated = False
            for other in corroborating or []:
                other_keys = {normalize_artist(p) for p in flatten_artist_names(other)}
                if len(part_keys & other_keys) >= 2:
                    corroborated = True
                    break
            if not corroborated and len(part_keys) >= 2:
                existing = 0
                for p in parts:
                    row = await db.fetch_one(
                        "SELECT 1 FROM artists WHERE artist_norm_imm(name) = artist_norm_imm(:n) LIMIT 1",
                        {"n": p},
                    )
                    if row:
                        existing += 1
                corroborated = existing >= 2
        chosen = parts if corroborated else [name]
        for p in chosen:
            if p and p not in names:
                names.append(p)
    return names or [n for n in raw_names if n]


async def ensure_entities_for_track(track_id: int) -> None:
    """Materialise artist/album rows for a track (rows only, not favorites).

    Needed so a liked track can navigate to its artist/album cards even when
    the track never appeared in a search that upserted those entities.
    Collab strings are resolved to participants first (evidence-gated), and
    EVERY participant gets a track_artists link — that is what makes the
    collab track show up on each participant's page.
    """
    track = await db.fetch_one(
        "SELECT title, artists, album, year, cover_url FROM tracks WHERE id = :t",
        {"t": track_id},
    )
    if not track:
        return
    names = [n for n in (track["artists"] or []) if n and n.strip()]
    if not names:
        return
    # Other sources of the same track may carry the artists split apart —
    # free corroboration for the evidence-gated splitter.
    meta_rows = await db.fetch_all(
        "SELECT artists FROM track_source_metadata WHERE track_id = :t",
        {"t": track_id},
    )
    corroborating = [list(r["artists"] or []) for r in meta_rows]
    resolved = await resolve_artist_names(names, corroborating)
    artist_ids = [await ensure_artist_by_name(name) for name in resolved]
    primary_artist = next((aid for aid in artist_ids if aid), None)
    linked = [int(aid) for aid in artist_ids if aid]
    if linked:
        await db.execute(
            """
            INSERT INTO track_artists (track_id, artist_id)
            SELECT :t, unnest(CAST(:ids AS bigint[]))
            ON CONFLICT DO NOTHING
            """,
            {"t": track_id, "ids": linked},
        )
    if track["album"] and primary_artist:
        await ensure_album_by_title(
            track["album"],
            primary_artist,
            track["year"],
            track["cover_url"],
        )


async def link_album_source(album_id: int, meta: AlbumMeta) -> None:
    """Attach a connector source to an existing album row (for lazy fetches)."""
    await _upsert_album_source(album_id, meta)


async def ensure_favorite_entities() -> int:
    """Backfill entity rows for liked tracks that still lack artist links.

    The old query walked every favorite on each API start (~10k statements,
    13-30 s of startup). Likes already run ensure_entities_for_track, so after
    the first pass this is empty.
    """
    rows = await db.fetch_all(
        """
        SELECT f.track_id
        FROM favorites f
        JOIN tracks t ON t.id = f.track_id
        WHERE cardinality(t.artists) > 0
          AND NOT EXISTS (
              SELECT 1 FROM track_artists ta WHERE ta.track_id = f.track_id
          )
        """
    )
    for r in rows:
        try:
            await ensure_entities_for_track(r["track_id"])
        except Exception:
            log.warning("ensure entities failed for track %s", r["track_id"], exc_info=True)
    return len(rows)
