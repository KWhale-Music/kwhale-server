"""Track upserts for lazy entity fetches (album tracklists, artist top tracks).

Runs the same dedup+upsert rules as search, without the search fan-out.
"""

from __future__ import annotations

from ..connectors.base import TrackMeta
from .search_service import upsert_track_metas


async def upsert_track_metas_bulk(metas: list[TrackMeta]) -> list[int]:
    return await upsert_track_metas(metas)
