"""ensure_seed_user is create-only: it must never reset an existing account.

Before the fix, every boot (and every test run) overwrote the seed user's
password_hash and forced role='admin', making password changes impossible and
silently promoting any invited user who happened to use the seed email.
"""

import pytest

from app import db
from app.auth import ensure_seed_user
from app.config import settings


async def _purge_seed() -> None:
    await db.execute(
        "DELETE FROM users WHERE lower(email) = lower(:e) OR username = :u",
        {"e": settings.seed_email, "u": settings.seed_username},
    )


@pytest.fixture(autouse=True)
async def _schema():
    await db.apply_schema()
    await _purge_seed()
    yield
    await _purge_seed()  # leave a clean slate even if the test fails
    await db.close_engine()


@pytest.mark.asyncio
async def test_ensure_seed_user_creates_admin_when_missing():
    await ensure_seed_user()
    row = await db.fetch_one(
        "SELECT role FROM users WHERE username = :u", {"u": settings.seed_username}
    )
    assert row is not None
    assert row["role"] == "admin"


@pytest.mark.asyncio
async def test_ensure_seed_user_never_resets_existing_password_or_role():
    await db.execute(
        "INSERT INTO users (username, password_hash, email, display_name, role) "
        "VALUES (:u, 'CUSTOMHASH', :e, 'Custom', 'user')",
        {"u": settings.seed_username, "e": settings.seed_email},
    )
    await ensure_seed_user()
    row = await db.fetch_one(
        "SELECT password_hash, role FROM users WHERE username = :u",
        {"u": settings.seed_username},
    )
    assert row["password_hash"] == "CUSTOMHASH", "пароль не должен сбрасываться"
    assert row["role"] == "user", "роль не должна повышаться до admin"
