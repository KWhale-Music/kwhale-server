"""Async DB layer: SQLAlchemy core engine + session factory.

Kept minimal: raw SQL through SQLAlchemy core for migrations-free startup
(schema.sql applied idempotently at boot), Core-style inserts/selects in
services. No ORM models to keep the schema visible in one file (schema.sql).
"""

from __future__ import annotations

import re
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from contextvars import ContextVar
from typing import Any

import sqlalchemy
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine, create_async_engine

from .config import settings

_engine: AsyncEngine | None = None

# Request-scoped write transaction. Inside `async with db.batch()` every helper
# below reuses one connection and commits once on exit instead of autocommit per
# statement. A single search used to issue ~1,400 COMMITs (~8 ms fsync each);
# this turns the persistence tail of /v1/search from ~11 s into ~0.1 s.
_batch_conn: ContextVar[AsyncConnection | None] = ContextVar("kwhale_db_batch", default=None)

_SCHEMA_PATH = __file__.replace("db.py", "schema.sql")


async def get_engine() -> AsyncEngine:
    global _engine
    if _engine is None:
        _engine = create_async_engine(
            settings.database_url,
            pool_pre_ping=True,
            pool_size=settings.db_pool_size,
            max_overflow=settings.db_max_overflow,
            # One pathological analytics query must not occupy a connection
            # for minutes and starve the API pool.
            connect_args={
                "server_settings": {"statement_timeout": str(settings.db_statement_timeout_ms)}
            },
        )
    return _engine


async def close_engine() -> None:
    global _engine
    if _engine is not None:
        await _engine.dispose()
        _engine = None


async def apply_schema() -> None:
    """Apply schema.sql (idempotent) once per process."""
    engine = await get_engine()
    # Drop pre-v2 materialized views before schema.sql runs so new view
    # definitions (with `repeats` and artist columns) never collide with old ones.
    async with engine.begin() as conn:
        await conn.execute(sqlalchemy.text(_DROP_STALE_MV_V1_DDL))
    with open(_SCHEMA_PATH) as f:
        raw = f.read()
    # split on semicolons that end a statement (naive but fine: no functions/DO blocks)
    statements = [s.strip() for s in raw.split(";") if s.strip()]
    async with engine.begin() as conn:
        for stmt in statements:
            await conn.execute(sqlalchemy.text(stmt))
    # columns added after first deploy (idempotent)
    for stmt in (
        "ALTER TABLE tracks ADD COLUMN IF NOT EXISTS year INT",
        "ALTER TABLE tracks ADD COLUMN IF NOT EXISTS track_number INT",
        "ALTER TABLE tracks ADD COLUMN IF NOT EXISTS genre TEXT",
        "ALTER TABLE tracks ADD COLUMN IF NOT EXISTS track_total INT",
        "ALTER TABLE tracks ADD COLUMN IF NOT EXISTS disc_number INT",
        "ALTER TABLE tracks ADD COLUMN IF NOT EXISTS disc_total INT",
        "ALTER TABLE tracks ADD COLUMN IF NOT EXISTS explicit BOOLEAN",
        "ALTER TABLE tracks ADD COLUMN IF NOT EXISTS identity_hash TEXT",
        "CREATE INDEX IF NOT EXISTS tracks_identity_hash ON tracks (identity_hash) WHERE identity_hash IS NOT NULL",
        "ALTER TABLE tracks ADD COLUMN IF NOT EXISTS has_lyrics BOOLEAN NOT NULL DEFAULT FALSE",
        "ALTER TABLE tracks ADD COLUMN IF NOT EXISTS genres_norm TEXT[] NOT NULL DEFAULT '{}'",
        "CREATE INDEX IF NOT EXISTS tracks_genres_norm ON tracks USING gin (genres_norm)",
        "ALTER TABLE albums ADD COLUMN IF NOT EXISTS explicit BOOLEAN NOT NULL DEFAULT FALSE",
        "ALTER TABLE albums ADD COLUMN IF NOT EXISTS album_type TEXT",
        "ALTER TABLE idempotency_keys DROP CONSTRAINT IF EXISTS idempotency_keys_pkey",
        "ALTER TABLE idempotency_keys ADD PRIMARY KEY (user_id, endpoint, key)",
        "ALTER TABLE track_features ADD COLUMN IF NOT EXISTS dsp_extra JSONB NOT NULL DEFAULT '{}'",
        "ALTER TABLE track_features ADD COLUMN IF NOT EXISTS subgenres TEXT[] NOT NULL DEFAULT '{}'",
        "ALTER TABLE track_features ADD COLUMN IF NOT EXISTS moods TEXT[] NOT NULL DEFAULT '{}'",
        "ALTER TABLE track_features ADD COLUMN IF NOT EXISTS textures TEXT[] NOT NULL DEFAULT '{}'",
        "ALTER TABLE track_features ADD COLUMN IF NOT EXISTS scenes TEXT[] NOT NULL DEFAULT '{}'",
        "ALTER TABLE track_features ADD COLUMN IF NOT EXISTS language TEXT",
        "ALTER TABLE track_features ADD COLUMN IF NOT EXISTS era TEXT",
        "ALTER TABLE track_features ADD COLUMN IF NOT EXISTS vibe_summary TEXT",
        "ALTER TABLE track_features ADD COLUMN IF NOT EXISTS semantic_energy REAL",
        "ALTER TABLE track_features ADD COLUMN IF NOT EXISTS semantic_valence REAL",
        "ALTER TABLE track_features ADD COLUMN IF NOT EXISTS acousticness REAL",
        "ALTER TABLE track_features ADD COLUMN IF NOT EXISTS instrumentalness REAL",
        "ALTER TABLE track_features ADD COLUMN IF NOT EXISTS tags_source TEXT",
        "ALTER TABLE track_features ADD COLUMN IF NOT EXISTS tags_version TEXT",
        "ALTER TABLE track_features ADD COLUMN IF NOT EXISTS tags_indexed_at TIMESTAMPTZ",
        "CREATE INDEX IF NOT EXISTS track_features_subgenres_gin ON track_features USING gin (subgenres)",
        "CREATE INDEX IF NOT EXISTS track_features_moods_gin ON track_features USING gin (moods)",
        "CREATE INDEX IF NOT EXISTS track_features_textures_gin ON track_features USING gin (textures)",
        "CREATE INDEX IF NOT EXISTS track_features_scenes_gin ON track_features USING gin (scenes)",
        "CREATE TABLE IF NOT EXISTS schema_flags (key TEXT PRIMARY KEY,"
        " applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW())",
    ):
        try:
            async with engine.begin() as conn:
                await conn.execute(sqlalchemy.text(stmt))
        except Exception:  # noqa: S110 — already exists
            pass
    # One-shot rebuilds, guarded by schema_flags. CREATE OR REPLACE of
    # artist_norm_imm() does NOT recompute the stored generated column
    # tracks.artists_norm nor the expression index artists_norm_imm_idx — they
    # keep values of the old function body and every artist-page prefilter
    # silently misses (the «Тёмный Принц» bug). Rebuild once per normalization
    # change: bump the flag key when the function body changes again.
    await _run_once(engine, "norm_parity_v1", _NORM_REBUILD_DDL)
    await _run_once(engine, "rec_mv_v2", _REC_MV_V2_STMTS)


_DROP_STALE_MV_V1_DDL = """
DO $mig$
BEGIN
    IF to_regclass('mv_user_track_stats') IS NOT NULL THEN
        IF NOT EXISTS (
            SELECT 1 FROM pg_attribute
            WHERE attrelid = to_regclass('mv_user_track_stats')
              AND attname = 'repeats'
              AND NOT attisdropped
        ) THEN
            DROP MATERIALIZED VIEW IF EXISTS mv_user_artist_stats;
            DROP MATERIALIZED VIEW IF EXISTS mv_user_track_stats;
            DROP MATERIALIZED VIEW IF EXISTS mv_item_cooccurrence;
        END IF;
    END IF;
END
$mig$
"""

_NORM_REBUILD_DDL = """
DO $mig$
BEGIN
    ALTER TABLE tracks DROP COLUMN IF EXISTS artists_norm;
    ALTER TABLE tracks ADD COLUMN artists_norm TEXT
        GENERATED ALWAYS AS (public.artist_norm_imm(public.artists_text_imm(artists))) STORED;
    CREATE INDEX IF NOT EXISTS tracks_artists_norm_trgm
        ON tracks USING gin (artists_norm gin_trgm_ops);
    IF to_regclass('artists_norm_imm_idx') IS NOT NULL THEN
        EXECUTE 'REINDEX INDEX artists_norm_imm_idx';
    END IF;
END
$mig$
"""

_REC_MV_V2_STMTS: tuple[str, ...] = (
    "DROP MATERIALIZED VIEW IF EXISTS mv_user_artist_stats",
    "DROP MATERIALIZED VIEW IF EXISTS mv_user_track_stats",
    "DROP MATERIALIZED VIEW IF EXISTS mv_item_cooccurrence",
    """
    CREATE MATERIALIZED VIEW mv_user_track_stats AS
    SELECT user_id,
           track_id,
           COUNT(*) FILTER (WHERE type = 'play') AS plays,
           COUNT(*) FILTER (
               WHERE type = 'complete'
                  OR (
                      type IN ('skip', 'pause')
                      AND (payload->>'listenedMs') ~ '^\\d+$'
                      AND (payload->>'trackDurationMs') ~ '^\\d+$'
                      AND (payload->>'trackDurationMs')::bigint > 0
                      AND (payload->>'listenedMs')::double precision
                          >= 0.85 * (payload->>'trackDurationMs')::double precision
                  )
           ) AS completes,
           COUNT(*) FILTER (
               WHERE type = 'skip'
                 AND NOT (
                     (payload->>'listenedMs') ~ '^\\d+$'
                     AND (payload->>'trackDurationMs') ~ '^\\d+$'
                     AND (payload->>'trackDurationMs')::bigint > 0
                     AND (payload->>'listenedMs')::double precision
                         >= 0.85 * (payload->>'trackDurationMs')::double precision
                 )
           ) AS skips,
           COUNT(*) FILTER (WHERE type = 'repeat') AS repeats,
           MAX(COALESCE(occurred_at, received_at)) FILTER (WHERE type IN ('play', 'complete', 'repeat')) AS last_played
    FROM events
    WHERE track_id IS NOT NULL
    GROUP BY user_id, track_id
    """,
    "CREATE UNIQUE INDEX mv_user_track_stats_pk ON mv_user_track_stats (user_id, track_id)",
    """
    CREATE MATERIALIZED VIEW mv_user_artist_stats AS
    SELECT s.user_id,
           an.artist_name,
           SUM(s.plays)     AS plays,
           SUM(s.completes) AS completes,
           SUM(s.skips)     AS skips,
           SUM(s.repeats)   AS repeats,
           COUNT(*) FILTER (WHERE f.track_id IS NOT NULL) AS liked
    FROM mv_user_track_stats s
    JOIN tracks t ON t.id = s.track_id
    CROSS JOIN LATERAL unnest(t.artists) AS an(artist_name)
    LEFT JOIN favorites f ON f.user_id = s.user_id AND f.track_id = s.track_id
    GROUP BY s.user_id, an.artist_name
    """,
    "CREATE UNIQUE INDEX mv_user_artist_stats_pk ON mv_user_artist_stats (user_id, artist_name)",
    """
    CREATE MATERIALIZED VIEW mv_item_cooccurrence AS
    WITH sess_events AS (
        SELECT DISTINCT
               e.user_id,
               e.track_id,
               floor(extract(epoch FROM COALESCE(e.occurred_at, e.received_at)) / 1800.0)::bigint AS win
        FROM events e
        WHERE e.track_id IS NOT NULL
          AND e.type IN ('play', 'complete', 'repeat', 'like')
          AND COALESCE(e.occurred_at, e.received_at) > NOW() - INTERVAL '180 days'
          AND NOT EXISTS (
              SELECT 1 FROM events sk
              WHERE sk.user_id = e.user_id
                AND sk.track_id = e.track_id
                AND sk.playback_id IS NOT NULL
                AND sk.playback_id = e.playback_id
                AND sk.type = 'skip'
                AND (sk.payload->>'listenedMs') ~ '^\\d+$'
                AND (sk.payload->>'listenedMs')::bigint < 20000
          )
    ),
    pairs AS (
        SELECT a.track_id AS track_a, b.track_id AS track_b, (COUNT(*) * 2.0)::double precision AS w
        FROM playlist_tracks a
        JOIN playlist_tracks b ON a.playlist_id = b.playlist_id AND a.track_id < b.track_id
        GROUP BY a.track_id, b.track_id
        UNION ALL
        SELECT a.track_id AS track_a, b.track_id AS track_b, COUNT(*)::double precision AS w
        FROM sess_events a
        JOIN sess_events b ON a.user_id = b.user_id AND a.win = b.win AND a.track_id < b.track_id
        GROUP BY a.track_id, b.track_id
    )
    SELECT track_a, track_b, SUM(w) AS weight
    FROM pairs
    GROUP BY track_a, track_b
    """,
    "CREATE UNIQUE INDEX mv_item_cooccurrence_pk ON mv_item_cooccurrence (track_a, track_b)",
)


async def _run_once(engine, flag: str, ddl: str | tuple[str, ...]) -> None:
    """Execute a non-idempotent DDL block exactly once per database."""
    try:
        async with engine.begin() as conn:
            done = await conn.execute(
                sqlalchemy.text("SELECT 1 FROM schema_flags WHERE key = :k"), {"k": flag}
            )
            if done.first() is not None:
                return
            stmts = (ddl,) if isinstance(ddl, str) else ddl
            for stmt in stmts:
                await conn.execute(sqlalchemy.text(stmt))
            await conn.execute(
                sqlalchemy.text("INSERT INTO schema_flags (key) VALUES (:k)"), {"k": flag}
            )
    except Exception:  # noqa: S110 — apply_schema must never block startup
        pass


@asynccontextmanager
async def batch() -> AsyncIterator[AsyncConnection]:
    """One transaction for a group of writes.

    Nested calls join the outer transaction. On success the transaction is
    committed once; on an unhandled exception it is rolled back as a whole.
    Use `savepoint()` around units that may fail independently.
    """
    current = _batch_conn.get()
    if current is not None:
        yield current
        return
    engine = await get_engine()
    async with engine.begin() as conn:
        token = _batch_conn.set(conn)
        try:
            yield conn
        finally:
            _batch_conn.reset(token)


@asynccontextmanager
async def savepoint() -> AsyncIterator[AsyncConnection | None]:
    """Per-unit isolation inside a batch (no-op outside a batch)."""
    conn = _batch_conn.get()
    if conn is None:
        yield None
        return
    async with conn.begin_nested():
        yield conn


@asynccontextmanager
async def get_conn() -> AsyncIterator[AsyncConnection]:
    current = _batch_conn.get()
    if current is not None:
        # Never close a connection owned by `batch()`.
        yield current
        return
    engine = await get_engine()
    async with engine.connect() as conn:
        yield conn


async def fetch_all(sql: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    async with get_conn() as conn:
        result = await conn.execute(sqlalchemy.text(sql), params or {})
        rows = result.mappings().all()
        return [dict(r) for r in rows]


async def fetch_one(sql: str, params: dict[str, Any] | None = None) -> dict[str, Any] | None:
    rows = await fetch_all(sql, params)
    return rows[0] if rows else None


async def execute(sql: str, params: dict[str, Any] | None = None) -> None:
    async with get_conn() as conn:
        await conn.execute(sqlalchemy.text(sql), params or {})
        if _batch_conn.get() is None:
            await conn.commit()


async def execute_many(sql: str, params: list[dict[str, Any]]) -> None:
    """Batch write (executemany) — one round-trip for many rows."""
    if not params:
        return
    async with get_conn() as conn:
        await conn.execute(sqlalchemy.text(sql), params)
        if _batch_conn.get() is None:
            await conn.commit()


async def execute_returning(sql: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    async with get_conn() as conn:
        result = await conn.execute(sqlalchemy.text(sql), params or {})
        if _batch_conn.get() is None:
            await conn.commit()
        return [dict(r) for r in result.mappings().all()]


def _quote_ident(name: str) -> str:
    if not re.fullmatch(r"[a-z_][a-z0-9_]*", name):
        raise ValueError(f"unsafe identifier: {name}")
    return f'"{name}"'


def insert_sql(table: str, cols: list[str], returning: str | None = None) -> str:
    """Build parameterised INSERT ... RETURNING for simple cases."""
    qcols = [_quote_ident(c) for c in cols]
    params = ", ".join(f":{c}" for c in cols)
    sql = f"INSERT INTO {_quote_ident(table)} ({', '.join(qcols)}) VALUES ({params})"
    if returning:
        sql += f" RETURNING {returning}"
    return sql
