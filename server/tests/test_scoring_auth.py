"""Scoring profile endpoints require auth (GET=user, PUT=admin).

Before the fix, GET/PUT /v1/scoring/profiles/{label} were fully anonymous, so
anyone on the network could read and rewrite the global recommendation ranker
parameters.
"""

import uuid

import pytest
from httpx import ASGITransport, AsyncClient

from app import db
from app.auth import ensure_seed_user, hash_password
from app.config import settings
from app.main import app


@pytest.fixture(autouse=True)
async def _schema():
    await db.apply_schema()
    await ensure_seed_user()
    yield
    await db.close_engine()


@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


async def _login(client, email, password):
    r = await client.post("/v1/auth/login", json={"email": email, "password": password})
    assert r.status_code == 200, r.text
    return {"Authorization": f"Bearer {r.json()['accessToken']}"}


async def _admin_headers(client):
    return await _login(client, settings.seed_email, settings.seed_password)


async def _user_headers(client):
    s = uuid.uuid4().hex[:8]
    email = f"scoring-{s}@example.com"
    await db.execute(
        "INSERT INTO users (username, password_hash, email, role) VALUES (:u, :p, :e, 'user')",
        {"u": f"scoring-{s}", "p": hash_password("pw"), "e": email},
    )
    return await _login(client, email, "pw")


@pytest.mark.asyncio
async def test_scoring_requires_auth(client):
    g = await client.get("/v1/scoring/profiles/default")
    assert g.status_code in (401, 403), "GET scoring без auth должен быть закрыт"
    p = await client.put("/v1/scoring/profiles/default", json={"valence_min": 0.5})
    assert p.status_code in (401, 403), "PUT scoring без auth должен быть закрыт"


@pytest.mark.asyncio
async def test_scoring_put_requires_admin(client):
    user_headers = await _user_headers(client)
    r = await client.put(
        "/v1/scoring/profiles/default", json={"valence_min": 0.5}, headers=user_headers
    )
    assert r.status_code == 403, "не-админ не должен менять скоринг"

    # reading is allowed for any authenticated user
    g = await client.get("/v1/scoring/profiles/default", headers=user_headers)
    assert g.status_code == 200

    # admin can write
    admin_headers = await _admin_headers(client)
    ok = await client.put(
        "/v1/scoring/profiles/default", json={"valence_min": 0.5}, headers=admin_headers
    )
    assert ok.status_code == 200, ok.text
