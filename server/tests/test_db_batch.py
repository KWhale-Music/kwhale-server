"""db.batch(): one transaction per request, savepoints for independent rows."""

from __future__ import annotations

import pytest

from app import db


@pytest.fixture(autouse=True)
async def _schema():
    await db.apply_schema()
    await db.execute("TRUNCATE tracks CASCADE")
    yield
    await db.close_engine()


@pytest.mark.asyncio
async def test_batch_commits_everything_on_success():
    async with db.batch():
        await db.execute("INSERT INTO tracks (title, artists) VALUES ('A', '{X}')")
        await db.execute("INSERT INTO tracks (title, artists) VALUES ('B', '{X}')")

    rows = await db.fetch_all("SELECT title FROM tracks ORDER BY title")
    assert [r["title"] for r in rows] == ["A", "B"]


@pytest.mark.asyncio
async def test_savepoint_rolls_back_only_failed_unit():
    async with db.batch():
        await db.execute("INSERT INTO tracks (title, artists) VALUES ('A', '{X}')")
        with pytest.raises(RuntimeError):
            async with db.savepoint():
                await db.execute("INSERT INTO tracks (title, artists) VALUES ('B', '{X}')")
                raise RuntimeError("one bad row")
        await db.execute("INSERT INTO tracks (title, artists) VALUES ('C', '{X}')")

    rows = await db.fetch_all("SELECT title FROM tracks ORDER BY title")
    assert [r["title"] for r in rows] == ["A", "C"]


@pytest.mark.asyncio
async def test_batch_rolls_back_on_unhandled_error():
    with pytest.raises(RuntimeError):
        async with db.batch():
            await db.execute("INSERT INTO tracks (title, artists) VALUES ('A', '{X}')")
            raise RuntimeError("abort the batch")

    rows = await db.fetch_all("SELECT title FROM tracks")
    assert rows == []
