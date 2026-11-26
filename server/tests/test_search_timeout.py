"""The second search wave must not outlive the client's read timeout."""

from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace

import pytest

from app import db
from app.services import search_service


@pytest.fixture(autouse=True)
async def _schema():
    await db.apply_schema()
    yield
    for task in list(search_service._INFLIGHT_BG_SEARCHES.values()):
        task.cancel()
    search_service._INFLIGHT_BG_SEARCHES.clear()
    await db.close_engine()


async def test_wait_all_returns_partial_before_deadline(monkeypatch):
    async def one_connector():
        fake = SimpleNamespace(search_albums=None, search_artists=None)
        return {"fake": fake}

    async def slow_search(*args, **kwargs):
        await asyncio.sleep(5)
        return []

    async def slow_section(*args, **kwargs):
        await asyncio.sleep(5)
        return []

    monkeypatch.setattr(search_service, "_active_connectors", one_connector)
    monkeypatch.setattr(search_service, "_connector_search", slow_search)
    monkeypatch.setattr(search_service, "_cached_section", slow_section)
    monkeypatch.setattr(search_service, "WAIT_ALL_TIMEOUT_SEC", 0.15)

    started = time.monotonic()
    result = await search_service.search(
        "deadline query", 5, user_id=None, wait_all=True
    )
    elapsed = time.monotonic() - started
    assert elapsed < 1.0
    assert result["partial"] is True
    assert result["tracks"]["items"] == []
