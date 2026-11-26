"""Library repair: consolidate duplicate favorites and revive missing sources.

The search/dedup pipeline can leave a favorite attached to a source-less
duplicate: track_sources has a unique (connector, external_id), so when the
same recording is represented twice only one row keeps the source. 82 favorites
on production hit this (71 have a sibling with sources, 11 have neither).

For source-less favorites the module is intentionally more forgiving than the
library merge: normalized title + artists alone are enough to move the favorite
to a sibling that actually has a source. The exact-duration duplicates were
already consolidated by merge_duplicates; what remains here is the user's own
favorite pointing at a dead row, so leaving it unplayable is worse than picking
the canonical recording with the same title/artists.

  * move a favorite to a same-title/artists sibling with sources;
  * otherwise retry the unavailable source once;
  * otherwise search the enabled connectors and attach/move a matching source;
  * never steal a source from a track with a different title/artists.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

from .. import db
from ..connectors import get_connector
from . import search_service
from .merge import _key, _repoint
from .search_rank import match_score

log = logging.getLogger("kwhale.repair")


async def _favorites_without_available() -> list[dict[str, Any]]:
    return await db.fetch_all(
        """
        SELECT f.user_id, f.track_id, t.title, t.artists, t.duration_sec
        FROM favorites f
        JOIN tracks t ON t.id = f.track_id
        WHERE NOT EXISTS (
            SELECT 1 FROM track_sources s
            WHERE s.track_id = f.track_id AND s.available
        )
        ORDER BY f.track_id, f.user_id
        """
    )


async def _source_candidates() -> dict[tuple[str, tuple[str, ...]], list[dict[str, Any]]]:
    rows = await db.fetch_all(
        """
        SELECT t.id, t.title, t.artists, t.duration_sec, count(s.id) AS sources
        FROM tracks t
        JOIN track_sources s ON s.track_id = t.id AND s.available
        GROUP BY t.id, t.title, t.artists, t.duration_sec
        """
    )
    grouped: dict[tuple[str, tuple[str, ...]], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[_key(row["title"], row["artists"])].append(row)
    return grouped


def _pick_candidate(
    favorite: dict[str, Any],
    candidates: list[dict[str, Any]],
) -> dict[str, Any] | None:
    eligible = [c for c in candidates if c["id"] != favorite["track_id"]]
    if not eligible:
        return None
    # most sources, then stable lowest id (same rule as merge_duplicates)
    return sorted(eligible, key=lambda c: (-int(c.get("sources") or 0), c["id"]))[0]


async def _move_favorite(favorite: dict[str, Any], canonical_id: int) -> None:
    """Re-point every reference from the orphan favorite track to the canonical."""
    orphan = int(favorite["track_id"])
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
        await _repoint(table, canonical_id, orphan)
    # All references moved; the duplicate row is now inert.
    await db.execute("DELETE FROM tracks WHERE id = :i", {"i": orphan})


async def _retry_unavailable(track_id: int) -> int:
    """One resolve attempt for sources marked unavailable; never loops."""
    sources = await db.fetch_all(
        """
        SELECT id, connector, external_id
        FROM track_sources
        WHERE track_id = :t AND available = FALSE
        """,
        {"t": track_id},
    )
    reactivated = 0
    for source in sources:
        connector = get_connector(source["connector"])
        if connector is None:
            continue
        try:
            info = await connector.resolve_stream(source["external_id"])
        except Exception as exc:  # noqa: BLE001 — a dead source stays dead
            log.info("repair resolve %s/%s failed: %s", source["connector"], source["external_id"], exc)
            continue
        if info and info.url:
            await db.execute(
                "UPDATE track_sources SET available = TRUE, updated_at = NOW() WHERE id = :i",
                {"i": source["id"]},
            )
            reactivated += 1
    return reactivated


async def _attach_source(track_id: int, meta: Any) -> str:
    """Attach a search result to the favorite unless that source belongs elsewhere."""
    existing = await db.fetch_one(
        """
        SELECT track_id FROM track_sources
        WHERE connector = :c AND external_id = :e
        LIMIT 1
        """,
        {"c": meta.connector, "e": meta.external_id},
    )
    if existing is not None:
        if int(existing["track_id"]) == track_id:
            # The search just proved this source still resolves, but the row
            # was left available=false by an earlier failed stream.
            await db.execute(
                """
                UPDATE track_sources
                SET available = TRUE, updated_at = NOW()
                WHERE connector = :c AND external_id = :e AND track_id = :t
                """,
                {"c": meta.connector, "e": meta.external_id, "t": track_id},
            )
            return "attached"
        return "conflict"
    await db.execute(
        """
        INSERT INTO track_sources
            (track_id, connector, external_id, quality_tier, mime, available, updated_at)
        VALUES (:t, :c, :e, :tier, NULL, TRUE, NOW())
        """,
        {
            "t": track_id,
            "c": meta.connector,
            "e": meta.external_id,
            "tier": int(meta.quality_tier or 0),
        },
    )
    await db.execute(
        """
        INSERT INTO track_source_metadata
            (track_id, connector, external_id, title, artists, album, year,
             track_number, track_total, disc_number, disc_total, genre,
             cover_url, isrc, duration_sec, quality_tier, updated_at)
        VALUES (:t, :c, :e, :title, :artists, :album, :year,
                :tn, :tt, :dn, :dt, :genre, :cover, :isrc, :duration, :tier, NOW())
        ON CONFLICT (track_id, connector, external_id) DO UPDATE SET
            title = EXCLUDED.title, artists = EXCLUDED.artists,
            album = EXCLUDED.album, year = EXCLUDED.year,
            duration_sec = EXCLUDED.duration_sec,
            quality_tier = EXCLUDED.quality_tier, updated_at = NOW()
        """,
        {
            "t": track_id,
            "c": meta.connector,
            "e": meta.external_id,
            "title": meta.title,
            "artists": meta.artists,
            "album": meta.album or None,
            "year": meta.year,
            "tn": meta.track_number,
            "tt": meta.track_total,
            "dn": meta.disc_number,
            "dt": meta.disc_total,
            "genre": meta.genre,
            "cover": meta.cover_url or None,
            "isrc": meta.isrc,
            "duration": meta.duration_sec or None,
            "tier": int(meta.quality_tier or 0),
        },
    )
    return "attached"


async def _search_and_attach(favorite: dict[str, Any]) -> bool:
    artists = favorite.get("artists") or []
    query = " ".join([favorite.get("title") or "", artists[0] if artists else ""]).strip()
    if not query:
        return False
    try:
        metas = await search_service._fan_out(query, 10)
    except Exception as exc:  # noqa: BLE001
        log.warning("repair search %r failed: %s", query, exc)
        return False
    fav_duration = favorite.get("duration_sec") or 0
    ranked = []
    for meta in metas:
        score = match_score(favorite.get("title") or "", meta.title, *meta.artists)
        if score < 0.55:
            continue
        if fav_duration and meta.duration_sec:
            if abs(fav_duration - meta.duration_sec) > 20:
                continue
        ranked.append((score, meta))
    for _score, meta in sorted(ranked, key=lambda item: -item[0]):
        status = await _attach_source(int(favorite["track_id"]), meta)
        if status == "attached":
            return True
        if status == "conflict":
            # The source belongs to another track. If that track carries the
            # same title/artists, it is the canonical row for this favorite:
            # revive the source and re-point the favorite instead of leaving
            # it unplayable (41037 «Шёлк» → 40639 is exactly this case).
            existing = await db.fetch_one(
                """
                SELECT s.track_id, t.title, t.artists
                FROM track_sources s
                JOIN tracks t ON t.id = s.track_id
                WHERE s.connector = :c AND s.external_id = :e
                LIMIT 1
                """,
                {"c": meta.connector, "e": meta.external_id},
            )
            if existing is None:
                continue
            if _key(existing["title"], existing["artists"]) != _key(
                favorite["title"], favorite["artists"]
            ):
                continue
            await db.execute(
                """
                UPDATE track_sources
                SET available = TRUE, updated_at = NOW()
                WHERE connector = :c AND external_id = :e
                """,
                {"c": meta.connector, "e": meta.external_id},
            )
            await _move_favorite(favorite, int(existing["track_id"]))
            return True
    return False


async def repair_favorites(
    *,
    dry_run: bool = False,
    limit: int | None = None,
) -> dict[str, Any]:
    rows = await _favorites_without_available()
    # One decision per track, not per user.
    favorites: dict[int, dict[str, Any]] = {}
    for row in rows:
        favorites.setdefault(int(row["track_id"]), row)
    items = list(favorites.values())
    if limit:
        items = items[: max(0, int(limit))]

    candidates_by_key = await _source_candidates()
    moved = 0
    reactivated = 0
    attached = 0
    unrepaired: list[dict[str, str]] = []
    actions: list[dict[str, Any]] = []

    for favorite in items:
        key = _key(favorite["title"], favorite["artists"])
        candidate = _pick_candidate(favorite, candidates_by_key.get(key, []))
        if candidate is not None:
            actions.append(
                {
                    "trackId": str(favorite["track_id"]),
                    "action": "move_to_sibling",
                    "to": str(candidate["id"]),
                    "title": favorite["title"],
                }
            )
            if not dry_run:
                await _move_favorite(favorite, int(candidate["id"]))
            moved += 1
            continue

        # No sibling: try to revive a source already attached to this track.
        unavailable = await db.fetch_one(
            """
            SELECT count(*) AS n FROM track_sources
            WHERE track_id = :t AND available = FALSE
            """,
            {"t": favorite["track_id"]},
        )
        if unavailable and int(unavailable["n"] or 0) > 0:
            actions.append(
                {
                    "trackId": str(favorite["track_id"]),
                    "action": "retry_unavailable",
                    "title": favorite["title"],
                }
            )
            if dry_run:
                continue
            got = await _retry_unavailable(int(favorite["track_id"]))
            reactivated += got
            if got:
                continue
            # Dead source: fall through to a fresh connector search.

        # Last resort: search enabled connectors and attach a matching source.
        actions.append(
            {
                "trackId": str(favorite["track_id"]),
                "action": "search_and_attach",
                "title": favorite["title"],
            }
        )
        if not dry_run and await _search_and_attach(favorite):
            attached += 1
        elif not dry_run:
            unrepaired.append(
                {"trackId": str(favorite["track_id"]), "title": favorite["title"]}
            )

    return {
        "favoritesWithoutSource": len(items),
        "moved": moved,
        "reactivated": reactivated,
        "attached": attached,
        "unrepaired": unrepaired,
        "actions": actions,
        "dry_run": dry_run,
    }
