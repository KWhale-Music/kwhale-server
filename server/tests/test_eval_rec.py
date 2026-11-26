"""Pure and DB evaluation helpers for the recommendation layer."""

from __future__ import annotations

import json
import uuid

import pytest

from app import db
from app.eval_rec import event_metrics, hit_rate, jaccard, mean_percentile


def test_hit_rate_counts_top_k():
    ranked = [5, 4, 3, 2, 1]
    assert hit_rate([5, 1], ranked, k=2) == 0.5  # only 5 is in top-2
    assert hit_rate([5, 1], ranked, k=5) == 1.0
    assert hit_rate([], ranked, k=5) == 0.0


def test_mean_percentile_orders_positions():
    ranked = [10, 20, 30, 40, 50]
    assert mean_percentile([10], ranked) == 0.0
    assert mean_percentile([50], ranked) == 1.0
    assert mean_percentile([10, 50], ranked) == 0.5
    assert mean_percentile([999], ranked) is None
    assert mean_percentile([], ranked) is None


def test_jaccard_helper():
    assert jaccard([], []) == 0.0
    assert jaccard([1, 2], [3, 4]) == 0.0
    assert jaccard([1, 2, 3], [2, 3, 4]) == 0.5
    assert jaccard([1, 2], [1, 2]) == 1.0


@pytest.mark.asyncio
async def test_event_metrics_reads_payload_surface():
    await db.apply_schema()
    await db.execute("TRUNCATE users, tracks, events CASCADE")
    try:
        urows = await db.execute_returning(
            "INSERT INTO users (username, password_hash) VALUES ('eval_u', 'h') RETURNING id"
        )
        uid = int(urows[0]["id"])
        for ev_type, surf in (
            ("play", "mix"),
            ("complete", "mix"),
            ("skip", "home"),
            ("play", "home"),
        ):
            await db.execute(
                "INSERT INTO events (id, user_id, type, payload) "
                "VALUES (:id, :u, :t, CAST(:p AS jsonb))",
                {"id": str(uuid.uuid4()), "u": uid, "t": ev_type, "p": json.dumps({"surface": surf})},
            )
        metrics = await event_metrics(days=30)
        assert metrics["mix"]["completes"] == 1
        assert metrics["mix"]["plays"] == 1
        assert metrics["home"]["skips"] == 1
    finally:
        await db.execute("TRUNCATE users, tracks, events CASCADE")
        await db.close_engine()
