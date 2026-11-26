"""Conservative duplicate-merge for the track library.

Several rows can exist for the *same recording* when providers report slightly
different durations (e.g. 6 rows of «Группа крови» — Кино). This merges those
clusters so a listener sees one track. Crucially it is conservative: it only
merges tracks with the SAME normalized title + artists AND a close duration
(± tolerance), so different covers («Heart Shaped Box» by Nirvana vs Neovaii) and
different versions (Demo / Live / 2013 Mix) are NEVER touched.

References (favorites / dislikes / playlist_tracks / track_sources / metadata /
events) are re-pointed to the canonical track with conflict-safe deletes first,
so nothing is lost and no unique constraint is violated.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

from .. import db
from .dedup import duration_close, normalize_artist, normalize_title
from .normalize import (
    album_title_key,
    compact_artist_key,
    extract_featured_artists,
    flatten_artist_names,
)

log = logging.getLogger("kwhale.merge")

_DURATION_TOL = 15  # seconds — same recording reported with slightly different lengths


def _key(title: str | None, artists: list[str] | None) -> tuple[str, tuple[str, ...]]:
    # Providers disagree on how to pack a collaboration into the artists
    # array ("A feat. B, C" in one string vs ["A", "B", "C"]) or put feats
    # into the title. Normalize both sources identically.
    clean_title, feats = extract_featured_artists(title or "")
    flat = flatten_artist_names([*(artists or []), *feats], loose=True)
    normalized = {
        compact_artist_key(a) or normalize_artist(a)
        for a in flat
        if (compact_artist_key(a) or normalize_artist(a))
    }
    return (normalize_title(clean_title), tuple(sorted(normalized)))


def _clusters(members: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Group same-key tracks into duration clusters (sorted, greedy ±tolerance)."""
    # Sort known durations first so zero-length rows attach cleanly only when safe
    members = sorted(members, key=lambda r: (0 if (r["duration_sec"] or 0) > 0 else 1, r["duration_sec"] or 0))
    clusters: list[list[dict[str, Any]]] = []
    for row in members:
        placed = False
        dur = row["duration_sec"] or 0
        for cluster in clusters:
            ref = next((m["duration_sec"] or 0 for m in cluster if (m["duration_sec"] or 0) > 0), 0)
            if duration_close(dur, ref, _DURATION_TOL):
                cluster.append(row)
                placed = True
                break
        if not placed:
            clusters.append([row])
    return clusters


async def _source_count(track_id: int) -> int:
    row = await db.fetch_one(
        "SELECT COUNT(*) AS n FROM track_sources WHERE track_id = :t", {"t": track_id}
    )
    return int(row["n"]) if row else 0


async def _repoint(table: str, canon: int, dupe: int) -> None:
    """Move rows from dupe -> canon, dropping rows that would violate a unique key."""
    if table in ("favorites", "dislikes"):
        await db.execute(
            f"DELETE FROM {table} WHERE track_id = :d AND user_id IN "
            f"(SELECT user_id FROM {table} WHERE track_id = :c)",
            {"d": dupe, "c": canon},
        )
        await db.execute(f"UPDATE {table} SET track_id = :c WHERE track_id = :d", {"c": canon, "d": dupe})
    elif table == "playlist_tracks":
        await db.execute(
            "DELETE FROM playlist_tracks WHERE track_id = :d AND playlist_id IN "
            "(SELECT playlist_id FROM playlist_tracks WHERE track_id = :c)",
            {"d": dupe, "c": canon},
        )
        await db.execute("UPDATE playlist_tracks SET track_id = :c WHERE track_id = :d", {"c": canon, "d": dupe})
    elif table in ("track_sources", "track_source_metadata"):
        if table == "track_sources":
            # Do not move a local file that the quality scan proved corrupt:
            # it would make the canonical track serve garbage with 206. The
            # file stays on disk and fsck quarantines it.
            await db.execute(
                """
                DELETE FROM track_sources s
                WHERE s.track_id = :d AND s.connector = 'local'
                  AND EXISTS (
                      SELECT 1 FROM track_quality q
                      WHERE q.local_source_id = s.id AND q.error IS NOT NULL
                  )
                """,
                {"d": dupe},
            )
        await db.execute(
            f"DELETE FROM {table} WHERE track_id = :d AND (connector, external_id) IN "
            f"(SELECT connector, external_id FROM {table} WHERE track_id = :c)",
            {"d": dupe, "c": canon},
        )
        await db.execute(f"UPDATE {table} SET track_id = :c WHERE track_id = :d", {"c": canon, "d": dupe})
    elif table == "index_jobs":
        await db.execute(
            "DELETE FROM index_jobs WHERE track_id = :d AND (kind, model_version) IN "
            "(SELECT kind, model_version FROM index_jobs WHERE track_id = :c)",
            {"d": dupe, "c": canon},
        )
        await db.execute("UPDATE index_jobs SET track_id = :c WHERE track_id = :d", {"c": canon, "d": dupe})
    else:  # events / track_enrichment_proposals — no unique conflict on track_id
        await db.execute(f"UPDATE {table} SET track_id = :c WHERE track_id = :d", {"c": canon, "d": dupe})


async def merge_duplicates(dry_run: bool = False) -> dict[str, Any]:
    rows = await db.fetch_all("SELECT id, title, artists, duration_sec FROM tracks")
    groups: dict[tuple[str, tuple[str, ...]], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        key = _key(r.get("title"), r.get("artists"))
        if key[0]:
            groups[key].append(r)

    merged = 0
    clusters_merged = 0
    for _key_name, members in groups.items():
        if len(members) < 2:
            continue
        for cluster in _clusters(members):
            if len(cluster) < 2:
                continue
            # canonical: most sources, then lowest id (stable)
            counts = {r["id"]: await _source_count(r["id"]) for r in cluster}
            scored = sorted(cluster, key=lambda r: (-counts[r["id"]], r["id"]))
            canon = scored[0]["id"]
            dupes = [r["id"] for r in cluster if r["id"] != canon]
            clusters_merged += 1
            if dry_run:
                merged += len(dupes)
                continue
            for dupe in dupes:
                for table in (
                    "favorites",
                    "dislikes",
                    "playlist_tracks",
                    "track_sources",
                    "track_source_metadata",
                    "events",
                    "track_enrichment_proposals",
                    "index_jobs",
                ):
                    try:
                        await _repoint(table, canon, dupe)
                    except Exception:  # noqa: BLE001
                        log.exception("repoint %s %s -> %s failed", table, dupe, canon)
                await db.execute("DELETE FROM tracks WHERE id = :i", {"i": dupe})
            merged += len(dupes)
            log.info("merged cluster -> canonical %s (dropped %s)", canon, dupes)

    return {"clusters": clusters_merged, "merged": merged, "dry_run": dry_run}


# ── Duplicate artists ────────────────────────────────────────────────────────
# The artists table holds several rows per display name (source-less rows
# materialised by ensure_artist_by_name next to connector rows; Latin/Cyrillic
# homoglyphs: «Xoliday Boy» vs «Хoliday Boy»). Search then shows «множество
# дублирующихся исполнителей», and each duplicate opens an empty artist page
# (0 tracks, no cover). Merge rows with the same identity key into the richest
# canonical row; references are re-pointed before the delete.


def _artist_key_name(name: str | None) -> str:
    return compact_artist_key(name or "") or normalize_artist(name or "")


async def _repoint_artist_refs(canon: int, dupe: int) -> None:
    """Move artist references dupe -> canon, conflict-safe."""
    await db.execute(
        "DELETE FROM track_artists WHERE artist_id = :d AND track_id IN "
        "(SELECT track_id FROM track_artists WHERE artist_id = :c)",
        {"d": dupe, "c": canon},
    )
    await db.execute(
        "UPDATE track_artists SET artist_id = :c WHERE artist_id = :d",
        {"c": canon, "d": dupe},
    )
    await db.execute(
        "DELETE FROM artist_sources WHERE artist_id = :d AND (connector, external_id) IN "
        "(SELECT connector, external_id FROM artist_sources WHERE artist_id = :c)",
        {"d": dupe, "c": canon},
    )
    await db.execute(
        "UPDATE artist_sources SET artist_id = :c WHERE artist_id = :d",
        {"c": canon, "d": dupe},
    )
    await db.execute(
        "DELETE FROM favorite_artists WHERE artist_id = :d AND user_id IN "
        "(SELECT user_id FROM favorite_artists WHERE artist_id = :c)",
        {"d": dupe, "c": canon},
    )
    await db.execute(
        "UPDATE favorite_artists SET artist_id = :c WHERE artist_id = :d",
        {"c": canon, "d": dupe},
    )
    await db.execute(
        "UPDATE albums SET artist_id = :c WHERE artist_id = :d",
        {"c": canon, "d": dupe},
    )
    await db.execute(
        "DELETE FROM artist_sync_state WHERE artist_id = :d AND EXISTS "
        "(SELECT 1 FROM artist_sync_state s WHERE s.artist_id = :c)",
        {"d": dupe, "c": canon},
    )
    await db.execute(
        "UPDATE artist_sync_state SET artist_id = :c WHERE artist_id = :d",
        {"c": canon, "d": dupe},
    )


async def merge_duplicate_artists(dry_run: bool = False) -> dict[str, Any]:
    rows = await db.fetch_all(
        """
        SELECT ar.id, ar.name, ar.cover_url,
               (SELECT COUNT(*) FROM track_artists ta WHERE ta.artist_id = ar.id) AS track_links,
               (SELECT COUNT(*) FROM artist_sources s WHERE s.artist_id = ar.id) AS srcs,
               (SELECT COUNT(*) FROM albums al WHERE al.artist_id = ar.id) AS albums
        FROM artists ar
        """
    )
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        key = _artist_key_name(r.get("name"))
        if key:
            groups[key].append(r)

    merged = 0
    clusters_merged = 0
    for key, members in groups.items():
        if len(members) < 2:
            continue
        # canonical: has a cover, then most content (links/sources/albums), then lowest id
        scored = sorted(
            members,
            key=lambda r: (
                0 if r.get("cover_url") else 1,
                -(int(r["track_links"] or 0) + int(r["srcs"] or 0) + int(r["albums"] or 0)),
                r["id"],
            ),
        )
        canon = scored[0]["id"]
        dupes = [r["id"] for r in scored[1:]]
        clusters_merged += 1
        if dry_run:
            merged += len(dupes)
            continue
        for dupe in dupes:
            try:
                await _repoint_artist_refs(canon, dupe)
            except Exception:  # noqa: BLE001
                log.exception("repoint artist %s -> %s failed", dupe, canon)
            await db.execute("DELETE FROM artists WHERE id = :i", {"i": dupe})
        merged += len(dupes)
        log.info("merged artist cluster %r -> canonical %s (dropped %s)", key, canon, dupes)

    return {"clusters": clusters_merged, "merged": merged, "dry_run": dry_run}


# ── Duplicate album rows ─────────────────────────────────────────────────────
# Sources re-release the same record with decorated titles («LOCA LA VIDA» vs
# «LOCA LA VIDA - Single»), and those two rows both land on the artist page —
# the «одна и та же песня/обложка повторяется» feedback. Rows with the same
# artist and the same title after version-junk stripping merge into one; the
# canonical keeps the cover and the most sources.


async def _repoint_album_refs(canon: int, dupe: int) -> None:
    await db.execute(
        "DELETE FROM album_sources WHERE album_id = :d AND (connector, external_id) IN "
        "(SELECT connector, external_id FROM album_sources WHERE album_id = :c)",
        {"d": dupe, "c": canon},
    )
    await db.execute(
        "UPDATE album_sources SET album_id = :c WHERE album_id = :d",
        {"c": canon, "d": dupe},
    )
    await db.execute(
        "DELETE FROM favorite_albums WHERE album_id = :d AND user_id IN "
        "(SELECT user_id FROM favorite_albums WHERE album_id = :c)",
        {"d": dupe, "c": canon},
    )
    await db.execute(
        "UPDATE favorite_albums SET album_id = :c WHERE album_id = :d",
        {"c": canon, "d": dupe},
    )


async def merge_duplicate_albums(dry_run: bool = False) -> dict[str, Any]:
    rows = await db.fetch_all(
        """
        SELECT al.id, al.title, al.artist_id, al.cover_url, al.year, al.track_count,
               (SELECT COUNT(*) FROM album_sources s WHERE s.album_id = al.id) AS srcs,
               (SELECT COUNT(*) FROM favorite_albums fa WHERE fa.album_id = al.id) AS favs
        FROM albums al
        """
    )
    groups: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        key = (int(r["artist_id"]), album_title_key(r.get("title")))
        if key[1]:
            groups[key].append(r)

    merged = 0
    clusters_merged = 0
    for _key, members in groups.items():
        if len(members) < 2:
            continue
        # Same title but different release years are different records (a 2019
        # single and a 2020 album called the same) — only bridge ±1 year drift
        # when both rows come from connector sources and don't look like a
        # single (<=2 tracks) vs album (>=5 tracks) collision.
        clusters: list[list[dict[str, Any]]] = []
        for r in members:
            placed = False
            cur_year = r.get("year")
            cur_tc = r.get("track_count")
            cur_srcs = int(r.get("srcs") or 0)
            for cluster in clusters:
                ref = cluster[0]
                ref_year = ref.get("year")
                ref_tc = ref.get("track_count")
                ref_srcs = int(ref.get("srcs") or 0)
                same_year = ref_year is None or cur_year is None or ref_year == cur_year
                drift_ok = (
                    ref_year is not None
                    and cur_year is not None
                    and abs(int(ref_year) - int(cur_year)) <= 1
                    and ref_srcs > 0
                    and cur_srcs > 0
                    and not (
                        ref_tc is not None
                        and cur_tc is not None
                        and ((ref_tc <= 2 and cur_tc >= 5) or (cur_tc <= 2 and ref_tc >= 5))
                    )
                )
                if same_year or drift_ok:
                    cluster.append(r)
                    placed = True
                    break
            if not placed:
                clusters.append([r])
        for cluster in clusters:
            if len(cluster) < 2:
                continue
            # canonical: has a cover, favorited, most sources, then lowest id
            scored = sorted(
                cluster,
                key=lambda r: (
                    0 if r.get("cover_url") else 1,
                    -int(r["favs"] or 0),
                    -int(r["srcs"] or 0),
                    r["id"],
                ),
            )
            canon = scored[0]["id"]
            dupes = [r["id"] for r in scored[1:]]
            clusters_merged += 1
            if dry_run:
                merged += len(dupes)
                continue
            for dupe in dupes:
                try:
                    await _repoint_album_refs(canon, dupe)
                except Exception:  # noqa: BLE001
                    log.exception("repoint album %s -> %s failed", dupe, canon)
                await db.execute("DELETE FROM albums WHERE id = :i", {"i": dupe})
            merged += len(dupes)
            log.info("merged album cluster -> canonical %s (dropped %s)", canon, dupes)

    return {"clusters": clusters_merged, "merged": merged, "dry_run": dry_run}


# ── Collab artist rows ───────────────────────────────────────────────────────
# Sources pack collabs inconsistently ("Тимати,MOT,ЕГОР КРИД" as ONE artist
# name), and those strings became artist rows — 1.5k of them on production.
# They pollute search, own no real discography and hide the participants.
# `repair --split-artists` re-points their links to the real participants
# (evidence-gated in resolve_artist_names: a name splits only when the parts
# are supported, so band names with commas ("Tyler, The Creator") stay whole)
# and deletes the shell row.


async def split_collab_artists(dry_run: bool = False) -> dict[str, Any]:
    from .entity_upsert import ensure_artist_by_name, resolve_artist_names

    rows = await db.fetch_all("SELECT id, name FROM artists")
    split_count = 0
    deleted = 0
    for r in rows:
        name = r["name"] or ""
        # Corroboration: per-source metadata of the shell's tracks often has
        # the same collab packed differently ("Тимати","MOT",... separately).
        meta_rows = await db.fetch_all(
            """
            SELECT DISTINCT m.artists FROM track_source_metadata m
            JOIN track_artists ta ON ta.track_id = m.track_id
            WHERE ta.artist_id = :i
            LIMIT 20
            """,
            {"i": int(r["id"])},
        )
        corroborating = [list(m["artists"] or []) for m in meta_rows]
        resolved = await resolve_artist_names([name], corroborating)
        if len(resolved) <= 1:
            continue
        split_count += 1
        if dry_run:
            deleted += 1
            continue
        participant_ids: list[int] = []
        for p in resolved:
            pid = await ensure_artist_by_name(p)
            if pid:
                participant_ids.append(pid)
        dupe = int(r["id"])
        participant_ids = [pid for pid in participant_ids if pid != dupe]
        if not participant_ids:
            continue
        canon = participant_ids[0]
        for pid in participant_ids:
            await db.execute(
                """
                INSERT INTO track_artists (track_id, artist_id)
                SELECT track_id, :p FROM track_artists WHERE artist_id = :d
                ON CONFLICT DO NOTHING
                """,
                {"p": pid, "d": dupe},
            )
        await db.execute(
            "DELETE FROM favorite_artists WHERE artist_id = :d AND user_id IN "
            "(SELECT user_id FROM favorite_artists WHERE artist_id = :c)",
            {"d": dupe, "c": canon},
        )
        await db.execute(
            "UPDATE favorite_artists SET artist_id = :c WHERE artist_id = :d",
            {"c": canon, "d": dupe},
        )
        await db.execute(
            "DELETE FROM artist_sources WHERE artist_id = :d AND (connector, external_id) IN "
            "(SELECT connector, external_id FROM artist_sources WHERE artist_id = :c)",
            {"d": dupe, "c": canon},
        )
        await db.execute(
            "UPDATE artist_sources SET artist_id = :c WHERE artist_id = :d",
            {"c": canon, "d": dupe},
        )
        await db.execute("DELETE FROM artist_sync_state WHERE artist_id = :d", {"d": dupe})
        await db.execute("UPDATE albums SET artist_id = :c WHERE artist_id = :d", {"c": canon, "d": dupe})
        await db.execute("DELETE FROM artists WHERE id = :i", {"i": dupe})
        deleted += 1
        log.info("split collab artist %r -> %s (deleted shell %s)", name, resolved, dupe)

    return {"split": split_count, "deleted": deleted, "dry_run": dry_run}
