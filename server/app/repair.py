"""CLI for the library repair: duplicate merge + source-less favorites.

    python -m app.repair --dry-run
    python -m app.repair --merge --favorites
    python -m app.repair --favorites --limit 20

Run with the API stopped for a large merge; the DB remains available.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from typing import Any

from . import db, httpcore
from .config import settings
from .services.identity import identity_hash
from .services.merge import (
    merge_duplicate_albums,
    merge_duplicate_artists,
    merge_duplicates,
    split_collab_artists,
)
from .services.normalize import derive_album_type
from .services.repair import repair_favorites


async def backfill_identity_hash(dry_run: bool = False) -> dict[str, Any]:
    """Fill tracks.identity_hash for rows created before the identity key."""
    rows = await db.fetch_all(
        "SELECT id, title, artists FROM tracks WHERE identity_hash IS NULL OR identity_hash = ''"
    )
    filled = 0
    for r in rows:
        h = identity_hash(r.get("title"), r.get("artists"))
        if not dry_run:
            await db.execute(
                "UPDATE tracks SET identity_hash = :h WHERE id = :i",
                {"h": h, "i": r["id"]},
            )
        filled += 1
    return {"filled": filled, "dry_run": dry_run}


async def backfill_album_types(dry_run: bool = False) -> dict[str, Any]:
    """Fill/repair albums.album_type for rows created before the single/EP split.

    Two passes: NULL/empty rows get a derived type; rows whose title carries an
    explicit marker (" - Single") but whose stored type contradicts it are
    re-derived too — the title marker outranks whatever the heuristic guessed.
    """
    rows = await db.fetch_all(
        """
        SELECT id, title, track_count, album_type FROM albums
        WHERE album_type IS NULL OR album_type = ''
           OR (title ~* '(\\(\\s*(single|ep)\\s*\\)|[-–—]\\s*(single|ep)\\s*$)'
               AND album_type NOT IN ('single', 'ep'))
        """
    )
    filled = 0
    for r in rows:
        at = derive_album_type(r.get("title"), r.get("track_count"))
        if at == r.get("album_type"):
            continue
        if not dry_run:
            await db.execute(
                "UPDATE albums SET album_type = :at WHERE id = :i",
                {"at": at, "i": r["id"]},
            )
        filled += 1
    return {"filled": filled, "dry_run": dry_run}


async def backfill_genres(dry_run: bool = False) -> dict[str, Any]:
    """Fill tracks.genres_norm from raw provider genre strings (new in 0.1.17).

    Raw genres are provider soup («Hip-Hop & Rap», «rusrap», «хип-хоп»), so
    every recommendation component matches canonical families instead — this
    is the one-time backfill for rows that predate the column.
    """
    from .services.genres import normalize_genres

    rows = await db.fetch_all(
        "SELECT id, genre FROM tracks WHERE genre IS NOT NULL AND genre <> '' "
        "AND (genres_norm IS NULL OR genres_norm = '{}')"
    )
    filled = 0
    for r in rows:
        families = normalize_genres([r["genre"]])
        if not families:
            continue
        if not dry_run:
            await db.execute(
                "UPDATE tracks SET genres_norm = :g WHERE id = :i",
                {"g": families, "i": r["id"]},
            )
        filled += 1
    return {"filled": filled, "dry_run": dry_run}


async def backfill_valence(dry_run: bool = False) -> dict[str, Any]:
    """Recompute track_features.valence for rows indexed before the heuristic.

    Same formula as indexer.extract_dsp — rows that got features while valence
    was still never written keep NULL and quietly disable every mood filter.
    """
    rows = await db.fetch_all(
        "SELECT track_id, energy, danceability, mode FROM track_features "
        "WHERE energy IS NOT NULL AND danceability IS NOT NULL"
    )
    updated = 0
    for r in rows:
        energy = float(r["energy"])
        dance = float(r["danceability"])
        mode_val = 1 if int(r.get("mode") or 0) == 1 else 0
        valence = round(
            max(
                0.0,
                min(
                    1.0,
                    0.5
                    + (0.25 if mode_val == 1 else -0.25)
                    + 0.15 * (energy - 0.5)
                    + 0.35 * (dance - 0.5),
                ),
            ),
            4,
        )
        if not dry_run:
            await db.execute(
                "UPDATE track_features SET valence = :v WHERE track_id = :i",
                {"v": valence, "i": r["track_id"]},
            )
        updated += 1
    return {"updated": updated, "dry_run": dry_run}


async def _majority_track_cover(album_title: str, artist: str | None) -> str | None:
    """Evidence for the real artwork: the album's own tracks' covers."""
    row = await db.fetch_one(
        """
        SELECT t.cover_url, COUNT(*) AS n
        FROM tracks t
        WHERE t.album IS NOT NULL
          AND artist_norm_imm(t.album) = artist_norm_imm(:title)
          AND (CAST(:artist AS text) IS NULL OR EXISTS (
              SELECT 1 FROM unnest(t.artists) AS an
              WHERE artist_matches(an, :artist)
          ))
          AND t.cover_url IS NOT NULL AND t.cover_url <> ''
        GROUP BY t.cover_url
        ORDER BY n DESC
        LIMIT 1
        """,
        {"title": album_title, "artist": artist},
    )
    return row["cover_url"] if row else None


async def covers_audit(fix: bool = False, dry_run: bool = False) -> dict[str, Any]:
    """Albums that ended up with someone else's artwork.

    Album merges and cover backfills left cover URLs shared by albums with
    unrelated titles (the «Shh...» report). The fix only trusts evidence: a
    suspicious album's cover is replaced with the majority cover of its own
    tracks when that majority points somewhere else. No guessing otherwise.
    """
    from .services.normalize import album_title_key

    rows = await db.fetch_all(
        "SELECT a.id, a.title, a.cover_url, ar.name AS artist_name "
        "FROM albums a LEFT JOIN artists ar ON ar.id = a.artist_id "
        "WHERE a.cover_url IS NOT NULL AND a.cover_url <> ''"
    )
    by_url: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_url.setdefault(row["cover_url"], []).append(row)

    suspicious: list[dict[str, Any]] = []
    for url, group in by_url.items():
        if len(group) < 2:
            continue
        keys = {album_title_key(r["title"]) for r in group}
        if len(keys) < 2:
            continue  # same album in different spellings — one cover is right
        for album in group:
            others = [g["title"] for g in group if g["id"] != album["id"]]
            suspicious.append(
                {
                    "album_id": album["id"],
                    "title": album["title"],
                    "artist": album["artist_name"],
                    "cover": url,
                    "shared_with": others,
                }
            )

    applied: list[dict[str, Any]] = []
    if fix and not dry_run:
        for item in suspicious:
            derived = await _majority_track_cover(item["title"], item["artist"])
            if derived and derived != item["cover"]:
                await db.execute(
                    "UPDATE albums SET cover_url = :c WHERE id = :i",
                    {"c": derived, "i": item["album_id"]},
                )
                applied.append(
                    {"album_id": item["album_id"], "from": item["cover"], "to": derived}
                )
    return {"suspicious": suspicious, "applied": applied, "dry_run": dry_run}


async def content_audit(fix: bool = False, dry_run: bool = False) -> dict[str, Any]:
    """Find local files whose measured duration contradicts the track row.

    The «body double»/«amy» mixup (2026-11-18): a mislabeled source downloaded
    a different recording into the managed library — track_quality already
    recorded 167 s for a 112 s row, but nothing ever compared the two.
    Rows with junk preview durations (canonical <= 40 s) are report-only: the
    row is the suspect there, not the file. ``fix`` re-queues each wrong-file
    track through the downloader, whose ingest now verifies content identity
    and quarantines the bad copy on success.
    """
    rows = await db.fetch_all(
        """
        SELECT q.track_id, q.file_path, q.duration_sec AS measured,
               t.title, t.artists, t.duration_sec AS canonical
        FROM track_quality q JOIN tracks t ON t.id = q.track_id
        WHERE q.duration_sec IS NOT NULL AND t.duration_sec IS NOT NULL
          AND abs(q.duration_sec - t.duration_sec) > 30
        ORDER BY q.track_id
        """
    )

    def _report(r: dict[str, Any]) -> dict[str, Any]:
        return {
            "track_id": r["track_id"],
            "title": r["title"],
            "artists": r["artists"],
            "canonical_sec": r["canonical"],
            "measured_sec": r["measured"],
            "file_path": r["file_path"],
        }

    wrong_files = [_report(r) for r in rows if (r["canonical"] or 0) > 40]
    suspect_rows = [_report(r) for r in rows if (r["canonical"] or 0) <= 40]
    requeued: list[dict[str, Any]] = []
    if fix and not dry_run:
        from .services.downloader import enqueue

        for r in wrong_files:
            job = await enqueue(int(r["track_id"]), force=True)
            requeued.append({"track_id": r["track_id"], "job_id": job})
    return {
        "wrong_files": wrong_files,
        "suspect_rows": suspect_rows,
        "requeued": requeued,
        "dry_run": dry_run,
    }


async def _cli() -> None:
    parser = argparse.ArgumentParser(description="KWhale library repair")
    parser.add_argument("--merge", action="store_true", help="merge duplicate tracks")
    parser.add_argument("--artists", action="store_true", help="merge duplicate artists")
    parser.add_argument("--albums", action="store_true", help="merge duplicate album rows")
    parser.add_argument("--album-types", action="store_true", help="backfill albums.album_type")
    parser.add_argument("--identity", action="store_true", help="backfill tracks.identity_hash")
    parser.add_argument("--genres", action="store_true", help="backfill tracks.genres_norm families")
    parser.add_argument("--valence", action="store_true", help="recompute track_features.valence")
    parser.add_argument("--split-artists", action="store_true", help="split collab artist rows")
    parser.add_argument("--favorites", action="store_true", help="repair source-less favorites")
    parser.add_argument(
        "--content-audit",
        action="store_true",
        help="report local files whose measured duration contradicts the track",
    )
    parser.add_argument(
        "--covers-audit",
        action="store_true",
        help="report albums whose cover is shared with unrelated titles",
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="with --content-audit: re-queue mismatched files for verified re-download",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="max favorites to repair")
    args = parser.parse_args()
    exclusive = (
        args.favorites,
        args.artists,
        args.albums,
        args.album_types,
        args.identity,
        args.genres,
        args.valence,
        args.split_artists,
        args.merge,
        args.content_audit,
        args.covers_audit,
    )
    do_merge = args.merge or not any(exclusive)
    do_artists = args.artists or not any(exclusive)
    do_albums = args.albums or not any(exclusive)
    do_album_types = args.album_types or not any(exclusive)
    do_identity = args.identity or not any(exclusive)
    do_split_artists = args.split_artists or not any(exclusive)
    do_favorites = args.favorites or not any(exclusive)

    # Full-table scans legitimately exceed the API's per-statement watchdog.
    object.__setattr__(settings, "db_statement_timeout_ms", 0)
    await db.apply_schema()
    await httpcore.load_proxy_settings()
    result: dict[str, Any] = {}
    try:
        if args.content_audit:
            result["content_audit"] = await content_audit(fix=args.fix, dry_run=args.dry_run)
        if args.covers_audit:
            result["covers_audit"] = await covers_audit(fix=args.fix, dry_run=args.dry_run)
        if args.genres:
            result["genres"] = await backfill_genres(dry_run=args.dry_run)
        if args.valence:
            result["valence"] = await backfill_valence(dry_run=args.dry_run)
        if do_identity:
            result["identity"] = await backfill_identity_hash(dry_run=args.dry_run)
        if do_split_artists:
            result["split_artists"] = await split_collab_artists(dry_run=args.dry_run)
        if do_merge:
            result["merge"] = await merge_duplicates(dry_run=args.dry_run)
        if do_artists:
            result["artists"] = await merge_duplicate_artists(dry_run=args.dry_run)
        if do_albums:
            result["albums"] = await merge_duplicate_albums(dry_run=args.dry_run)
        if do_album_types:
            result["album_types"] = await backfill_album_types(dry_run=args.dry_run)
        if do_favorites:
            result["favorites"] = await repair_favorites(
                dry_run=args.dry_run,
                limit=args.limit,
            )
    finally:
        print(json.dumps(result, ensure_ascii=False, default=str))
        await httpcore.aclose_all()
        await db.close_engine()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(_cli())
