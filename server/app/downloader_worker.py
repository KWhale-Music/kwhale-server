"""Dedicated downloader process.

The download queue is I/O-heavy (HLS segments, tagging, verification) and used
to share the API event loop, which made searches slow while a bulk re-download
was running. Production starts this module in its own container
(`python -m app.downloader_worker`) and sets `DOWNLOADER_ENABLED=false` for the
API. The runner polls Postgres every 30 s, so jobs enqueued by the API are
picked up without any cross-process wake-up.
"""

from __future__ import annotations

import asyncio
import logging

from . import httpcore
from .services import downloader

log = logging.getLogger("kwhale.downloader_worker")


async def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    # Resolve SOURCE_PROXY_* exactly like the API does before any connector use.
    await httpcore.load_proxy_settings()
    await downloader.start_runner()
    log.info("downloader worker up")
    # The runner task owns the loop; keep the process alive until SIGTERM.
    await asyncio.Event().wait()


if __name__ == "__main__":
    asyncio.run(main())
