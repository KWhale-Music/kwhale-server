"""Test config: isolated test database — NEVER the live beta DB.

Tests point DATABASE_URL at a dedicated `kwhale_test` database (created on
demand) so a pytest run can never pollute the production `kwhale` DB
(users, source_accounts, playlists, ...). Override the target with
KWHALE_TEST_DATABASE_URL (e.g. for CI or an external test server).
"""

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy.engine.url import make_url  # noqa: E402

# --- target test DB ---------------------------------------------------------
_DEFAULT_TEST_URL = "postgresql+asyncpg://kwhale:kwhale_v3_pg@localhost:5433/kwhale_test"
_TEST_URL = os.environ.get("KWHALE_TEST_DATABASE_URL", _DEFAULT_TEST_URL)

# Safety guard: never run the suite against the live production database.
_URL_DB = make_url(_TEST_URL).database or ""
if _URL_DB == "kwhale" and os.environ.get("ALLOW_PROD_DB_TESTS") != "1":
    raise RuntimeError(
        "ОТКАЗ: тесты нацелены на боевую БД 'kwhale'. "
        "Используйте тестовую БД (kwhale_test) через KWHALE_TEST_DATABASE_URL "
        "или задайте ALLOW_PROD_DB_TESTS=1 осознанно (НЕ рекомендуется)."
    )

# Force (override any ambient DATABASE_URL) so tests cannot reach the live DB.
os.environ["DATABASE_URL"] = _TEST_URL
os.environ.setdefault("MUSIC_DIR", "/tmp/kwhale-test-music")
os.environ.setdefault("JWT_SECRET", "test-secret")
# Deterministic test-only seed credentials (the real SEED_PASSWORD lives in .env).
os.environ.setdefault("SEED_PASSWORD", "test-seed-password")
os.environ.setdefault("SEED_EMAIL", "seed-test@example.com")


def _ensure_test_db() -> None:
    """Prepare the test database (connects to the 'postgres' maint DB).

    For the default scratch DB `kwhale_test`, drop + recreate it each run so
    tests always start from a deterministic, clean slate (no stale seed users or
    leftover rows). For any custom KWHALE_TEST_DATABASE_URL we only create it if
    missing — never drop a database we are not sure is scratch.
    """
    import asyncpg

    u = make_url(_TEST_URL)

    async def _run() -> None:
        conn = await asyncpg.connect(
            host=u.host or "localhost",
            port=u.port or 5433,
            user=u.username,
            password=u.password,
            database="postgres",
            timeout=10,
        )
        try:
            if u.database == "kwhale_test":
                await conn.execute(f'DROP DATABASE IF EXISTS "{u.database}" WITH (FORCE)')
                await conn.execute(f'CREATE DATABASE "{u.database}"')
            else:
                exists = await conn.fetchval(
                    "SELECT 1 FROM pg_database WHERE datname = $1", u.database
                )
                if not exists:
                    await conn.execute(f'CREATE DATABASE "{u.database}"')
        finally:
            await conn.close()

    asyncio.run(_run())


_ensure_test_db()
