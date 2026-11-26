"""Source priorities: search/streaming vs canonical metadata.

Search/remote-streaming order is local > ICM > quality.
Canonical metadata order is editable data in source_metadata_priority.
"""

from __future__ import annotations

import time

from .. import db

# metadata_order() is read once per persisted track. A search with 100 tracks
# used to issue 100 identical SELECTs (~0.5 s); it is a tiny, rarely changed
# table, so a short process-local TTL is enough.
_METADATA_ORDER_TTL_SEC = 60.0
_metadata_order_cache: tuple[float, list[str]] | None = None

DEFAULT_METADATA_ORDER = [
    "icm",
    "yandex",
    "qobuz",
    "tidal",
    "deezer",
    "zvuk",
    "vk",
    "ytmusic",
    "soundcloud",
]


async def metadata_order() -> list[str]:
    global _metadata_order_cache
    now = time.monotonic()
    cached = _metadata_order_cache
    if cached is not None and now - cached[0] < _METADATA_ORDER_TTL_SEC:
        return cached[1]
    rows = await db.fetch_all(
        "SELECT connector FROM source_metadata_priority ORDER BY priority, connector"
    )
    order = [r["connector"] for r in rows] or DEFAULT_METADATA_ORDER
    _metadata_order_cache = (now, order)
    return order


async def metadata_rank(connector: str) -> int:
    order = await metadata_order()
    try:
        return order.index(connector)
    except ValueError:
        return len(order) + 1


def search_rank(connector: str, quality_tier: int) -> tuple[int, int]:
    """Local is not a TrackMeta; ICM is first among remote, then quality."""
    return (0 if connector == "icm" else 1, -int(quality_tier or 0))


async def set_metadata_order(order: list[str]) -> list[str]:
    global _metadata_order_cache
    clean: list[str] = []
    for name in order:
        name = name.strip().lower()
        if name and name not in clean:
            clean.append(name)
    if not clean:
        raise ValueError("order must not be empty")
    for idx, name in enumerate(clean, start=1):
        await db.execute(
            """
            INSERT INTO source_metadata_priority (connector, priority)
            VALUES (:c, :p)
            ON CONFLICT (connector) DO UPDATE SET priority = EXCLUDED.priority
            """,
            {"c": name, "p": idx * 10},
        )
    _metadata_order_cache = None
    return await metadata_order()


async def priority_rows() -> list[dict[str, object]]:
    rows = await db.fetch_all(
        "SELECT connector, priority FROM source_metadata_priority ORDER BY priority, connector"
    )
    return [{"connector": r["connector"], "priority": r["priority"]} for r in rows]
