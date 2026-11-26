"""CLI entry point for the local-library audio quality scanner.

    python -m app.quality --scan
    python -m app.quality --scan --force --limit 500
    python -m app.quality --track-id 123

The implementation lives in services/quality.py so the admin API and the
maintenance cycle share exactly the same code path.
"""

from __future__ import annotations

import asyncio
import logging

from .services.quality import _cli

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(_cli())
