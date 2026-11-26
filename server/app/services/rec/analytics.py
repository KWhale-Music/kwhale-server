"""Read-only recommendation analytics for the admin UI."""

from __future__ import annotations

from typing import Any

from ... import db
from .queue import MODEL_VERSIONS


def _iso(value: Any) -> str | None:
    if value is None:
        return None
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


async def overview() -> dict[str, Any]:
    features = await db.fetch_one(
        """
        SELECT count(*) AS total,
               count(*) FILTER (WHERE bpm IS NOT NULL) AS with_bpm,
               count(*) FILTER (WHERE energy IS NOT NULL) AS with_energy,
               count(*) FILTER (WHERE onset_rate IS NOT NULL) AS with_full_dsp,
               count(*) FILTER (WHERE cardinality(subgenres) > 0 OR cardinality(moods) > 0) AS with_tags,
               count(*) FILTER (WHERE tags_source = 'llm') AS llm_tagged,
               max(indexed_at) AS last_indexed
        FROM track_features
        """
    )
    local = await db.fetch_one(
        """
        SELECT count(*) AS total
        FROM tracks t
        WHERE EXISTS (
            SELECT 1 FROM track_sources s
            WHERE s.track_id = t.id AND s.connector = 'local' AND s.available
        )
        """
    )
    embeddings = await db.fetch_one(
        """
        SELECT count(*) AS total,
               count(*) FILTER (WHERE model_version = :v) AS v2_total,
               max(indexed_at) AS last_indexed
        FROM track_emb_text
        """,
        {"v": MODEL_VERSIONS["text"]},
    )
    audio_embs = await db.fetch_one(
        """
        SELECT count(*) AS total,
               count(*) FILTER (WHERE model_version = :v) AS v2_total,
               max(indexed_at) AS last_indexed
        FROM track_emb_audio
        """,
        {"v": MODEL_VERSIONS["audio"]},
    )
    jobs = await db.fetch_all(
        """
        SELECT kind, status, count(*) AS n
        FROM index_jobs
        GROUP BY kind, status
        ORDER BY kind, status
        """
    )
    profiles = await db.fetch_all(
        "SELECT label, version, updated_at FROM scoring_profiles ORDER BY label"
    )
    event_rows = await db.fetch_all(
        """
        SELECT type, count(*) AS n, max(occurred_at) AS last_event
        FROM events
        GROUP BY type
        ORDER BY n DESC
        """
    )
    weekly = await db.fetch_all(
        """
        SELECT w.user_id, w.playlist_id, w.updated_at,
               (SELECT count(*) FROM playlist_tracks pt WHERE pt.playlist_id = w.playlist_id) AS tracks
        FROM weekly_release_playlists w
        ORDER BY w.user_id
        """
    )
    mixes = await db.fetch_one(
        """
        SELECT count(*) AS n FROM playlists
        WHERE source = 'recommendation' AND is_system = TRUE
        """
    )

    return {
        "features": {
            "total": int((features or {}).get("total") or 0),
            "local": int((local or {}).get("total") or 0),
            "withBpm": int((features or {}).get("with_bpm") or 0),
            "withEnergy": int((features or {}).get("with_energy") or 0),
            "withFullDsp": int((features or {}).get("with_full_dsp") or 0),
            "withTags": int((features or {}).get("with_tags") or 0),
            "llmTagged": int((features or {}).get("llm_tagged") or 0),
            "lastIndexed": _iso((features or {}).get("last_indexed")),
        },
        "embeddings": {
            "total": int((embeddings or {}).get("total") or 0),
            "v2Total": int((embeddings or {}).get("v2_total") or 0),
            "lastIndexed": _iso((embeddings or {}).get("last_indexed")),
            "model": MODEL_VERSIONS["text"],
        },
        "audioEmbeddings": {
            "total": int((audio_embs or {}).get("total") or 0),
            "v2Total": int((audio_embs or {}).get("v2_total") or 0),
            "lastIndexed": _iso((audio_embs or {}).get("last_indexed")),
            "model": MODEL_VERSIONS["audio"],
        },
        "indexJobs": [
            {"kind": r["kind"], "status": r["status"], "count": int(r["n"])}
            for r in jobs
        ],
        "profiles": [
            {
                "label": r["label"],
                "version": int(r["version"]),
                "updatedAt": _iso(r.get("updated_at")),
            }
            for r in profiles
        ],
        "events": {
            "total": sum(int(r["n"]) for r in event_rows),
            "byType": [
                {
                    "type": r["type"],
                    "count": int(r["n"]),
                    "lastEvent": _iso(r.get("last_event")),
                }
                for r in event_rows
            ],
        },
        "weekly": [
            {
                "userId": int(r["user_id"]),
                "playlistId": int(r["playlist_id"]) if r.get("playlist_id") else None,
                "tracks": int(r.get("tracks") or 0),
                "updatedAt": _iso(r.get("updated_at")),
            }
            for r in weekly
        ],
        "recommendationPlaylists": int((mixes or {}).get("n") or 0),
    }
