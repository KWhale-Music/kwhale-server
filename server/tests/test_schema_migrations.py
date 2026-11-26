"""apply_schema must evolve EXISTING databases, not just create fresh ones.

The prod startup crash (2026-11-17): `tracks_identity_hash` index lived in
schema.sql next to CREATE TABLE, but on a deployed DB that CREATE is a no-op —
the column arrives later via ALTER in db.apply_schema's migration list. Fresh
test databases created the table WITH the column, so tests were green while
production was down. This test reproduces the deployed shape: drop the column
(so the table looks pre-identity), re-apply the schema, expect survival.
"""

import pytest
from sqlalchemy import text

from app import db


@pytest.fixture(autouse=True)
async def _schema():
    await db.apply_schema()
    yield
    await db.close_engine()


@pytest.mark.asyncio
async def test_apply_schema_evolves_existing_tracks_table():
    # Recreate the pre-identity state: the table exists, the column doesn't.
    await db.execute("DROP INDEX IF EXISTS tracks_identity_hash")
    await db.execute("ALTER TABLE tracks DROP COLUMN IF EXISTS identity_hash")

    # The exact prod path: applying the schema again must add the column and
    # index via migrations instead of failing on a stale CREATE INDEX.
    await db.apply_schema()

    row = await db.fetch_one(
        """
        SELECT column_name FROM information_schema.columns
        WHERE table_name = 'tracks' AND column_name = 'identity_hash'
        """
    )
    assert row is not None, "миграция должна добавить identity_hash в существующую таблицу"

    idx = await db.fetch_one(
        """
        SELECT indexname FROM pg_indexes
        WHERE tablename = 'tracks' AND indexname = 'tracks_identity_hash'
        """
    )
    assert idx is not None, "индекс identity_hash должен создаваться миграцией после ALTER"

    # and the column is usable for inserts right away
    await db.execute(
        "INSERT INTO tracks (title, artists, identity_hash, last_seen_at) "
        "VALUES ('mig', ARRAY['a'], 'x', NOW())"
    )
    await db.execute("DELETE FROM tracks WHERE title = 'mig'")


@pytest.mark.asyncio
async def test_apply_schema_is_idempotent():
    await db.apply_schema()
    await db.apply_schema()
    engine = await db.get_engine()
    async with engine.connect() as conn:
        result = await conn.execute(text("SELECT 1"))
        assert result.scalar() == 1


@pytest.mark.asyncio
async def test_apply_schema_evolves_existing_tracks_table_with_genres_norm():
    """The 2026-11-19 startup crash: genres_norm index lived in schema.sql.

    On a deployed DB the column arrives via the ALTER migration, so the index
    must be created after it — recreate the pre-migration shape and re-apply.
    """
    await db.execute("DROP INDEX IF EXISTS tracks_genres_norm")
    await db.execute("ALTER TABLE tracks DROP COLUMN IF EXISTS genres_norm")

    await db.apply_schema()

    column = await db.fetch_one(
        "SELECT column_name FROM information_schema.columns"
        " WHERE table_name = 'tracks' AND column_name = 'genres_norm'"
    )
    assert column is not None, "миграция должна добавить genres_norm"
    index = await db.fetch_one(
        "SELECT indexname FROM pg_indexes"
        " WHERE tablename = 'tracks' AND indexname = 'tracks_genres_norm'"
    )
    assert index is not None, "индекс genres_norm создаётся после ALTER"


@pytest.mark.asyncio
async def test_norm_rebuild_repairs_stale_generated_column_and_index():
    """The «Тёмный Принц» fix must reach DEPLOYED databases too.

    CREATE OR REPLACE of artist_norm_imm() does not recompute the stored
    generated column tracks.artists_norm nor the expression index
    artists_norm_imm_idx. Reproduce that shape: old function body computes a
    stale value, then apply_schema() must rebuild both and leave a once-only
    marker so later startups skip the work.
    """
    await db.execute("DELETE FROM schema_flags WHERE key = 'norm_parity_v1'")
    # old (pre-parity) body: no homoglyph folding, unaccent folds «ё» to Cyrillic «е»
    await db.execute(
        """
        CREATE OR REPLACE FUNCTION artist_norm_imm(text) RETURNS text
        LANGUAGE sql IMMUTABLE PARALLEL SAFE AS $$
            SELECT lower(regexp_replace(translate(unaccent($1), 'йЙ', 'иИ'), '[^\\w]', '', 'g'))
        $$
        """
    )
    rows = await db.execute_returning(
        "INSERT INTO tracks (title, artists, album, duration_sec, last_seen_at) "
        "VALUES ('mig-norm', ARRAY['Тёмный Принц'], '', 1, NOW()) RETURNING id"
    )
    tid = rows[0]["id"]
    stale = await db.fetch_one("SELECT artists_norm FROM tracks WHERE id = :i", {"i": tid})
    assert stale["artists_norm"] != "temhыипpиhц", "тест должен начинать со stale-значения"

    await db.apply_schema()

    fresh = await db.fetch_one("SELECT artists_norm FROM tracks WHERE id = :i", {"i": tid})
    assert fresh["artists_norm"] == "temhыипpиhц", "колонка должна пересчитаться миграцией"

    flag = await db.fetch_one("SELECT 1 FROM schema_flags WHERE key = 'norm_parity_v1'")
    assert flag is not None, "после пересчёта должен стоять маркер norm_parity_v1"

    idx = await db.fetch_one(
        "SELECT i.indisvalid AS ok FROM pg_index i"
        " WHERE i.indexrelid = to_regclass('artists_norm_imm_idx')"
    )
    assert idx is not None and idx["ok"] is True, "expression-индекс должен быть переиндексирован"

    await db.execute("DELETE FROM tracks WHERE title = 'mig-norm'")


@pytest.mark.asyncio
async def test_apply_schema_evolves_pre_v2_track_features_and_mvs():
    """Verify rec v2 ALTER columns, GIN indexes, and MV rebuild on an existing DB."""
    await db.execute("DELETE FROM schema_flags WHERE key = 'rec_mv_v2'")
    await db.execute("DROP INDEX IF EXISTS track_features_subgenres_gin")
    await db.execute("DROP INDEX IF EXISTS track_features_textures_gin")
    await db.execute("ALTER TABLE track_features DROP COLUMN IF EXISTS subgenres")
    await db.execute("ALTER TABLE track_features DROP COLUMN IF EXISTS textures")
    await db.execute("ALTER TABLE track_features DROP COLUMN IF EXISTS dsp_extra")
    await db.execute("DROP MATERIALIZED VIEW IF EXISTS mv_user_artist_stats")
    await db.execute("DROP MATERIALIZED VIEW IF EXISTS mv_user_track_stats")
    await db.execute(
        """
        CREATE MATERIALIZED VIEW mv_user_track_stats AS
        SELECT user_id, track_id, 0::bigint AS plays, 0::bigint AS completes,
               0::bigint AS skips, NULL::timestamptz AS last_played
        FROM events WHERE FALSE
        """
    )

    await db.apply_schema()

    cols = {
        r["column_name"]
        for r in await db.fetch_all(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name = 'track_features'"
        )
    }
    assert {"subgenres", "textures", "dsp_extra"} <= cols

    gin_idx = await db.fetch_one(
        "SELECT indexname FROM pg_indexes "
        "WHERE tablename = 'track_features' AND indexname = 'track_features_subgenres_gin'"
    )
    assert gin_idx is not None

    mv_cols = {
        r["attname"]
        for r in await db.fetch_all(
            "SELECT attname FROM pg_attribute "
            "WHERE attrelid = 'mv_user_track_stats'::regclass AND attnum > 0"
        )
    }
    assert "repeats" in mv_cols

    flag = await db.fetch_one("SELECT 1 FROM schema_flags WHERE key = 'rec_mv_v2'")
    assert flag is not None


@pytest.mark.asyncio
async def test_artist_norm_functions_work_with_empty_search_path():
    """pg_dump restores run under search_path='': generated columns must inline cleanly."""
    engine = await db.get_engine()
    async with engine.begin() as conn:
        await conn.execute(text("SET LOCAL search_path = ''"))
        row = (
            await conn.execute(
                text(
                    "SELECT public.artist_norm('Тёмный Принц') AS a,"
                    " public.artist_norm_imm('Тёмный Принц') AS b,"
                    " public.artist_matches('Madk1d & Тёмный Принц', 'Тёмный Принц') AS m"
                )
            )
        ).one()
        assert row.a == "temhыипpиhц"
        assert row.b == "temhыипpиhц"
        assert row.m is True

        await conn.execute(
            text(
                "CREATE TEMP TABLE tmp_norm_restore_check ("
                " artists text[] NOT NULL,"
                " artists_norm text GENERATED ALWAYS AS"
                " (public.artist_norm_imm(public.artists_text_imm(artists))) STORED"
                ")"
            )
        )
        await conn.execute(
            text("INSERT INTO tmp_norm_restore_check (artists) VALUES (ARRAY['Тёмный Принц'])")
        )
        gen = (
            await conn.execute(text("SELECT artists_norm FROM tmp_norm_restore_check"))
        ).scalar_one()
        assert gen == "temhыипpиhц"
