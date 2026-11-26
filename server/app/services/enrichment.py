"""Metadata enrichment proposals for manual uploads.

After a user uploads a file manually, we search the connected sources for the
same title/artist and, when a confident match is found, propose richer metadata
(ISRC / album / year / genre / cover). The user reviews and confirms the proposal
from the profile before anything is written to the track — nothing is applied
silently.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from .. import db
from . import search_service
from .dedup import normalize_artist, normalize_title

log = logging.getLogger("kwhale.enrichment")

_PROPOSABLE = ("isrc", "album", "year", "genre", "cover_url")


async def _track(track_id: int) -> dict[str, Any] | None:
    return await db.fetch_one(
        "SELECT id, title, artists, album, year, genre FROM tracks WHERE id = :i",
        {"i": track_id},
    )


async def _match_source(track: dict[str, Any]) -> dict[str, Any] | None:
    """Search the sources and return the best matching track row (metadata donor)."""
    title = track.get("title") or ""
    if not title:
        return None
    artists = track.get("artists") or []
    query = f"{title} {', '.join(artists)}".strip()
    try:
        res = await search_service.search(query, 10, None, wait_all=False)
    except Exception:
        log.exception("enrichment search failed for track %s", track["id"])
        return None

    nt = normalize_title(title)
    wanted = {normalize_artist(a) for a in artists if normalize_artist(a)}
    items = ((res or {}).get("tracks") or {}).get("items") or []
    for cand in items:
        cid = cand.get("id")
        if cid is None or int(cid) == int(track["id"]):
            continue
        cm = await _track(int(cid))
        if not cm or normalize_title(cm.get("title") or "") != nt:
            continue
        have = {normalize_artist(a) for a in (cm.get("artists") or []) if normalize_artist(a)}
        if wanted & have:
            return cm
    return None


async def enrich_track(track_id: int, user_id: int) -> int | None:
    """Build a metadata proposal for a manually-uploaded track. Returns proposal id."""
    track = await _track(track_id)
    if not track:
        return None
    donor = await _match_source(track)
    if not donor:
        return None

    proposed: dict[str, Any] = {}
    # propose only fields the uploaded track is missing (never overwrite blindly)
    for field in ("album", "year", "genre"):
        if not track.get(field) and donor.get(field):
            proposed[field] = donor[field]

    # ISRC lives in track_source_metadata of the matched source track
    if not await db.fetch_one(
        "SELECT isrc FROM track_source_metadata WHERE track_id = :i AND isrc IS NOT NULL AND isrc <> '' LIMIT 1",
        {"i": track_id},
    ):
        row = await db.fetch_one(
            "SELECT isrc FROM track_source_metadata WHERE track_id = :i "
            "AND isrc IS NOT NULL AND isrc <> '' LIMIT 1",
            {"i": donor["id"]},
        )
        if row and row.get("isrc"):
            proposed["isrc"] = row["isrc"]

    if not proposed:
        return None

    rows = await db.execute_returning(
        """
        INSERT INTO track_enrichment_proposals (track_id, user_id, proposed, source, status)
        VALUES (:t, :u, CAST(:p AS jsonb), 'sources', 'pending')
        RETURNING id
        """,
        {"t": track_id, "u": user_id, "p": json.dumps(proposed)},
    )
    log.info("enrichment proposal %s for track %s: %s", rows[0]["id"], track_id, list(proposed))
    return rows[0]["id"]


async def list_pending(user_id: int) -> list[dict[str, Any]]:
    rows = await db.fetch_all(
        """
        SELECT p.id, p.track_id, p.proposed, p.source, p.created_at,
               t.title, t.artists
        FROM track_enrichment_proposals p
        JOIN tracks t ON t.id = p.track_id
        WHERE p.user_id = :u AND p.status = 'pending'
        ORDER BY p.created_at DESC
        """,
        {"u": user_id},
    )
    out = []
    for r in rows:
        proposed = r.get("proposed") or {}
        if isinstance(proposed, str):
            try:
                proposed = json.loads(proposed)
            except ValueError:
                proposed = {}
        out.append(
            {
                "id": r["id"],
                "trackId": r["track_id"],
                "title": r["title"],
                "artists": r["artists"],
                "proposed": proposed,
                "source": r.get("source"),
                "createdAt": r["created_at"].isoformat() if r.get("created_at") else None,
            }
        )
    return out


async def _get(proposal_id: int, user_id: int) -> dict[str, Any] | None:
    return await db.fetch_one(
        "SELECT * FROM track_enrichment_proposals WHERE id = :i AND user_id = :u",
        {"i": proposal_id, "u": user_id},
    )


async def confirm(proposal_id: int, user_id: int) -> bool:
    p = await _get(proposal_id, user_id)
    if not p or p["status"] != "pending":
        return False
    proposed = p.get("proposed") or {}
    if isinstance(proposed, str):
        try:
            proposed = json.loads(proposed)
        except ValueError:
            proposed = {}
    allowed = {k: v for k, v in proposed.items() if k in _PROPOSABLE}
    if allowed:
        # merge into track + track_source_metadata
        set_cols = []
        params: dict[str, Any] = {"t": p["track_id"]}
        for field in ("album", "year", "genre"):
            if field in allowed:
                set_cols.append(f"{field} = :{field}")
                params[field] = allowed[field]
        if set_cols:
            await db.execute(
                f"UPDATE tracks SET {', '.join(set_cols)} WHERE id = :t", params
            )
        if "cover_url" in allowed:
            await db.execute(
                "UPDATE track_sources SET cover_url = :c WHERE track_id = :t",
                {"c": allowed["cover_url"], "t": p["track_id"]},
            )
        if "isrc" in allowed:
            await db.execute(
                """
                INSERT INTO track_source_metadata (track_id, connector, external_id, isrc, updated_at)
                SELECT :t, 'local', 'local::enriched', :isrc, NOW()
                WHERE NOT EXISTS (
                    SELECT 1 FROM track_source_metadata WHERE track_id = :t AND isrc IS NOT NULL AND isrc <> ''
                )
                ON CONFLICT (track_id, connector, external_id) DO UPDATE SET isrc = EXCLUDED.isrc
                """,
                {"t": p["track_id"], "isrc": allowed["isrc"]},
            )
    await db.execute(
        "UPDATE track_enrichment_proposals SET status = 'confirmed', updated_at = NOW() WHERE id = :i",
        {"i": proposal_id},
    )
    return True


async def reject(proposal_id: int, user_id: int) -> bool:
    p = await _get(proposal_id, user_id)
    if not p or p["status"] != "pending":
        return False
    await db.execute(
        "UPDATE track_enrichment_proposals SET status = 'rejected', updated_at = NOW() WHERE id = :i",
        {"i": proposal_id},
    )
    return True
