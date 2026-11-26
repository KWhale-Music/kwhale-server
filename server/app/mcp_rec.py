"""MCP server: recommendation controls for an LLM agent.

Run as a stdio MCP server:

    python -m app.mcp_rec

The LLM never ranks tracks itself. It lists mixes, builds them from a prompt
(the prompt only patches scoring weights, clamped by `validate_patch`), and
can read or update named scoring profiles.
"""

from __future__ import annotations

import json
from typing import Any

from fastmcp import FastMCP

from . import db
from .services.rec import llm as rec_llm
from .services.rec import mixes as rec_mixes
from .services.rec.stats import refresh_stats as _refresh_stats

mcp = FastMCP("kwhale-recommendations")


def _row(item: Any) -> dict:
    candidate = item.candidate
    return {
        "trackId": candidate.track_id,
        "artist": candidate.artist,
        "genre": candidate.genre,
        "score": getattr(item, "score", None),
        "reasonCode": getattr(item, "reason_code", None),
        "reason": rec_mixes.reason_text(item),  # type: ignore[arg-type]
    }


@mcp.tool()
async def list_mixes(user_id: int = 1) -> list[dict]:
    """List available mixes for a user (personal, discovery, genres, artists)."""
    return await rec_mixes.list_mixes(user_id)


@mcp.tool()
async def mix_tracks(mix_id: str, limit: int = 20, user_id: int = 1) -> list[dict]:
    """Ranked tracks for a mix id: personal, discovery, genre:<name>, artist:<id>, radio:<trackId>."""
    scored = await rec_mixes.build(user_id, mix_id, max(1, min(limit, 100)), 0)
    return [_row(item) for item in scored]


@mcp.tool()
async def prompt_mix(prompt: str, size: int = 20, user_id: int = 1) -> dict:
    """Build a mix from a free-text prompt. Returns the applied scoring patch and the tracks."""
    patch = await rec_llm.patch_for(prompt)
    scored = await rec_mixes.prompt_mix(user_id, prompt, max(10, min(size, 100)))
    return {"patch": patch, "tracks": [_row(item) for item in scored]}


@mcp.tool()
async def explain_prompt_mix(prompt: str, size: int = 20, user_id: int = 1) -> dict:
    """Debug tool: inspect the QueryPlan and full candidate features/tags for a prompt mix."""
    plan = await rec_llm.plan_for(prompt)
    scored = await rec_mixes.prompt_mix(user_id, prompt, max(10, min(size, 100)))
    tracks = []
    for item in scored:
        c = item.candidate
        tracks.append(
            {
                **_row(item),
                "bpm": c.bpm,
                "energy": c.energy,
                "valence": c.valence,
                "acousticness": c.acousticness,
                "subgenres": c.subgenres,
                "moods": c.moods,
                "textures": c.textures,
                "scenes": c.scenes,
                "semQuerySim": c.sem_query_sim,
            }
        )
    return {"plan": plan.model_dump(), "tracks": tracks}


@mcp.tool()
async def similar_tracks(track_id: int, limit: int = 20, user_id: int = 1) -> list[dict]:
    """Tracks similar to the given track: playlist co-occurrence plus same artist or genre."""
    scored = await rec_mixes.similar(user_id, track_id, max(1, min(limit, 100)))
    return [_row(item) for item in scored]


@mcp.tool()
async def get_scoring_profile(label: str = "default") -> dict:
    """Read a named scoring profile. Unknown labels return an empty patch."""
    row = await db.fetch_one(
        "SELECT params, version FROM scoring_profiles WHERE label = :l",
        {"l": label},
    )
    return {
        "label": label,
        "params": (row or {}).get("params") or {},
        "version": (row or {}).get("version"),
    }


@mcp.tool()
async def set_scoring_profile(label: str, patch: dict) -> dict:
    """Create or update a scoring profile. The patch is clamped and normalised."""
    params = rec_llm.validate_patch(patch)
    profile = rec_llm.to_profile(params)
    await db.execute(
        """
        INSERT INTO scoring_profiles (label, params, version, updated_at)
        VALUES (:l, CAST(:p AS jsonb), 1, NOW())
        ON CONFLICT (label) DO UPDATE
        SET params = EXCLUDED.params,
            version = scoring_profiles.version + 1,
            updated_at = NOW()
        """,
        {"l": label, "p": json.dumps(profile)},
    )
    rec_mixes.clear_cache()
    return {"label": label, "params": profile}


@mcp.tool()
async def refresh_stats() -> str:
    """Recompute the materialized recommendation aggregates."""
    await _refresh_stats()
    rec_mixes.clear_cache()
    return "ok"


if __name__ == "__main__":
    mcp.run()
