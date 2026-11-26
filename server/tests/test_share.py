"""Share system tests: canonical TTL links for track/album/artist, Monet OG cards, web UI & GC."""

from __future__ import annotations

import io
from datetime import datetime

import pytest
from httpx import ASGITransport, AsyncClient
from PIL import Image

from app import auth, db
from app.config import settings
from app.main import app
from app.services import gc as gc_service
from app.services import share_card, share_service
from app.v1 import images


@pytest.fixture(autouse=True)
async def _schema():
    auth.clear_user_cache()
    share_card._card_cache.clear()
    share_card._palette_cache.clear()
    await db.apply_schema()
    await auth.ensure_seed_user()
    yield
    auth.clear_user_cache()
    await db.close_engine()


@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest.fixture
async def admin_headers(client):
    r = await client.post(
        "/v1/auth/login",
        json={"email": settings.seed_email, "password": settings.seed_password},
    )
    assert r.status_code == 200
    return {"Authorization": f"Bearer {r.json()['accessToken']}"}


async def _seed_entities() -> dict[str, int]:
    await db.execute("DELETE FROM share_clicks")
    await db.execute("DELETE FROM share_links")
    await db.execute("DELETE FROM track_sources")
    await db.execute("DELETE FROM album_sources")
    await db.execute("DELETE FROM artist_sources")
    await db.execute("DELETE FROM tracks")
    await db.execute("DELETE FROM albums")
    await db.execute("DELETE FROM artists")
    await db.execute("DELETE FROM source_accounts")
    for conn, enabled, status in (
        ("yandex", True, "connected"),
        ("icm", True, "connected"),
        ("vk", True, "connected"),
        ("deezer", False, "disabled"),
    ):
        await db.execute(
            """
            INSERT INTO source_accounts (connector, kind, status, enabled)
            VALUES (:c, 'account', :s, :e)
            """,
            {"c": conn, "s": status, "e": enabled},
        )

    art = await db.execute_returning(
        """
        INSERT INTO artists (name, cover_url, genre, last_seen_at)
        VALUES ('Молчат Дома', 'https://avatars.yandex.net/test-artist.jpg', 'Пост-панк', NOW())
        RETURNING id
        """
    )
    artist_id = int(art[0]["id"])
    # Deliberately mix a valid Yandex ID with synthetic `custom_artist_*` ICM ID
    await db.execute(
        """
        INSERT INTO artist_sources (artist_id, connector, external_id, available)
        VALUES (:a, 'yandex', '5965630', TRUE),
               (:a, 'icm', 'custom_artist_1780260460_95f01978', TRUE)
        """,
        {"a": artist_id},
    )

    alb = await db.execute_returning(
        """
        INSERT INTO albums (title, artist_id, year, track_count, cover_url, last_seen_at)
        VALUES ('Этажи', :a, 2018, 9, 'https://avatars.yandex.net/test-album.jpg', NOW())
        RETURNING id
        """,
        {"a": artist_id},
    )
    album_id = int(alb[0]["id"])
    await db.execute(
        """
        INSERT INTO album_sources (album_id, connector, external_id, available)
        VALUES (:al, 'icm', '1484820476', TRUE),
               (:al, 'yandex', 'album:synthetic', TRUE)
        """,
        {"al": album_id},
    )

    trk = await db.execute_returning(
        """
        INSERT INTO tracks (title, artists, album, duration_sec, cover_url, last_seen_at)
        VALUES ('Звёзды', '{Молчат Дома}', 'Этажи', 230, 'https://avatars.yandex.net/test-track.jpg', NOW())
        RETURNING id
        """
    )
    track_id = int(trk[0]["id"])
    await db.execute(
        """
        INSERT INTO track_sources (track_id, connector, external_id, available)
        VALUES (:t, 'icm', '1484820482', TRUE),
               (:t, 'yandex', '0bdc5abc-uuid-invalid', TRUE),
               (:t, 'vk', '-2000123_456789', TRUE)
        """,
        {"t": track_id},
    )

    # Pre-seed cover image data in Postgres so card renderer produces real cover composition
    img_buf = io.BytesIO()
    Image.new("RGB", (64, 64), (42, 118, 196)).save(img_buf, format="PNG")
    img_bytes = img_buf.getvalue()
    for url in (
        "https://avatars.yandex.net/test-track.jpg",
        "https://avatars.yandex.net/test-album.jpg",
        "https://avatars.yandex.net/test-artist.jpg",
    ):
        iid = images.image_id_for_url(url)
        await db.execute(
            """
            INSERT INTO images (id, source_url, mime, data, width, height, fetched_at, created_at, last_accessed_at)
            VALUES (:id, :u, 'image/png', :d, 64, 64, NOW(), NOW(), NOW())
            ON CONFLICT (id) DO UPDATE SET data = EXCLUDED.data, mime = EXCLUDED.mime
            """,
            {"id": iid, "u": url, "d": img_bytes},
        )
    return {"artist_id": artist_id, "album_id": album_id, "track_id": track_id}


@pytest.mark.asyncio
async def test_share_track_album_artist_canonical_and_connected_sources(client, admin_headers):
    ids = await _seed_entities()

    # 1. Track share: canonical ID reuse + sliding TTL
    r1 = await client.post(
        "/v1/share",
        json={"kind": "track", "id": str(ids["track_id"])},
        headers=admin_headers,
    )
    assert r1.status_code == 200
    d1 = r1.json()
    assert d1["kind"] == "track"
    assert d1["deepLink"] == f"kwhale://track/{ids['track_id']}"
    assert d1["ttlDays"] == 7
    assert "/share/s/" in d1["shareUrl"]

    # Connected services only (deezer is disabled and absent on track -> must NOT appear)
    by_plat = {p["platform"]: p for p in d1["platforms"]}
    assert "deezer" not in by_plat
    assert by_plat["apple"]["exact"] is True
    assert by_plat["apple"]["url"] == "https://music.apple.com/song/1484820482"
    assert by_plat["vk"]["exact"] is True
    assert by_plat["vk"]["url"] == "https://vk.com/audio-2000123_456789"
    # Yandex had UUID -> safe fallback to search
    assert by_plat["yandex"]["exact"] is False
    assert by_plat["yandex"]["url"].startswith("https://music.yandex.ru/search?")

    # Second call for the same track reuses the same short id
    r2 = await client.post(
        "/v1/share",
        json={"kind": "track", "id": str(ids["track_id"])},
        headers=admin_headers,
    )
    assert r2.status_code == 200
    d2 = r2.json()
    assert d2["id"] == d1["id"]
    assert datetime.fromisoformat(d2["expiresAt"]) >= datetime.fromisoformat(d1["expiresAt"])

    # 2. Album share: valid ICM album ID vs synthetic `album:*` Yandex fallback
    ra = await client.post(
        "/v1/share",
        json={"kind": "album", "id": str(ids["album_id"])},
        headers=admin_headers,
    )
    assert ra.status_code == 200
    da = ra.json()
    assert da["kind"] == "album"
    assert da["deepLink"] == f"kwhale://album/{ids['album_id']}"
    ap = {p["platform"]: p for p in da["platforms"]}
    assert ap["apple"]["url"] == "https://music.apple.com/album/1484820476"
    assert ap["apple"]["exact"] is True
    assert ap["yandex"]["exact"] is False

    # 3. Artist share: valid Yandex artist ID vs synthetic `custom_artist_*` Apple fallback
    rart = await client.post(
        "/v1/share",
        json={"kind": "artist", "id": str(ids["artist_id"])},
        headers=admin_headers,
    )
    assert rart.status_code == 200
    dart = rart.json()
    assert dart["kind"] == "artist"
    assert dart["deepLink"] == f"kwhale://artist/{ids['artist_id']}"
    arp = {p["platform"]: p for p in dart["platforms"]}
    assert arp["yandex"]["url"] == "https://music.yandex.ru/artist/5965630"
    assert arp["yandex"]["exact"] is True
    assert arp["apple"]["exact"] is False


@pytest.mark.asyncio
async def test_share_public_web_og_card_cover_and_clicks(client, admin_headers):
    ids = await _seed_entities()
    created = (
        await client.post(
            "/v1/share",
            json={"kind": "album", "id": str(ids["album_id"])},
            headers={**admin_headers, "X-Forwarded-Proto": "https", "X-Forwarded-Host": "music.dueattendant149.org"},
        )
    ).json()
    sid = created["id"]
    assert created["shareUrl"] == f"https://music.dueattendant149.org/share/s/{sid}"

    # Web preview page (public, M3 adaptive HTML + deep link + OG tags, edge-cacheable)
    page = await client.get(
        f"/share/s/{sid}",
        headers={"X-Forwarded-Proto": "https", "X-Forwarded-Host": "music.dueattendant149.org"},
    )
    assert page.status_code == 200
    assert "public" in (page.headers.get("cache-control") or "")
    html_body = page.text
    assert 'og:image" content="https://music.dueattendant149.org/share/card/' in html_body
    assert f'href="kwhale://album/{ids["album_id"]}"' in html_body
    assert "Альбом" in html_body
    assert "Этажи" in html_body
    assert "<md-filled-button" in html_body
    assert "<md-filled-tonal-button" in html_body
    assert "<md-list>" in html_body
    assert "<md-list-item" in html_body
    assert "/share/static/m3-share.js" in html_body
    assert "/share/static/kwhale-logo.png" in html_body
    assert "backdrop-art" not in html_body
    assert ">K<" not in html_body

    # Bundled M3 JS, fonts, KWhale orca logo, and all 9 parsed brand SVGs must be served with traversal protection
    js_res = await client.get("/share/static/m3-share.js")
    assert js_res.status_code == 200
    assert "javascript" in js_res.headers["content-type"]
    for pf in ("yandex", "apple", "ytmusic", "soundcloud", "vk", "deezer", "tidal", "qobuz", "zvuk"):
        svg_res = await client.get(f"/share/static/icons/{pf}.svg")
        assert svg_res.status_code == 200
        assert "<svg" in svg_res.text
    assert (await client.get("/share/static/kwhale-logo.png")).status_code == 200
    assert (await client.get("/share/static/../config.py")).status_code == 404

    # 1200x630 PNG & JPG OG card + ETag 304
    card_png = await client.get(f"/share/card/{sid}.png")
    assert card_png.status_code == 200
    assert card_png.headers["content-type"] == "image/png"
    assert "public" in card_png.headers["cache-control"]
    with Image.open(io.BytesIO(card_png.content)) as im:
        assert im.size == (1200, 630)
    etag = card_png.headers["etag"]
    assert (await client.get(f"/share/card/{sid}.png", headers={"If-None-Match": etag})).status_code == 304

    card_jpg = await client.get(f"/share/card/{sid}.jpg")
    assert card_jpg.status_code == 200
    assert card_jpg.headers["content-type"] == "image/jpeg"

    # Cover endpoint + immutable header
    cover = await client.get(f"/share/cover/{sid}")
    assert cover.status_code == 200
    assert "public" in cover.headers["cache-control"]

    # Click tracking redirect must be no-store and increment DB counters
    go = await client.get(f"/share/go/{sid}/apple", follow_redirects=False)
    assert go.status_code == 302
    assert go.headers["location"] == "https://music.apple.com/album/1484820476"
    assert go.headers["cache-control"] == "no-store"

    details = await client.get(f"/v1/share/{sid}", headers=admin_headers)
    assert details.status_code == 200
    assert details.json()["clicksTotal"] == 1
    click_count = await db.fetch_one(
        "SELECT COUNT(*) AS n FROM share_clicks WHERE share_id = :id AND platform = 'apple'",
        {"id": sid},
    )
    assert int((click_count or {})["n"]) == 1


@pytest.mark.asyncio
async def test_share_ttl_expiry_410_rotation_and_gc(client, admin_headers):
    ids = await _seed_entities()
    r1 = await client.post("/v1/share", json={"kind": "track", "id": str(ids["track_id"])}, headers=admin_headers)
    sid1 = r1.json()["id"]
    img_id = r1.json()["imageId"]

    # Age the cached cover beyond image_cache_ttl_days while share link is STILL active -> GC must NOT delete it!
    await db.execute(
        "UPDATE images SET last_accessed_at = NOW() - INTERVAL '30 days' WHERE id = :i",
        {"i": img_id},
    )
    stats1 = await gc_service.run_gc()
    assert stats1["shares_deleted"] == 0
    assert await db.fetch_one("SELECT 1 FROM images WHERE id = :i", {"i": img_id}) is not None

    # Now expire the share link -> endpoints must return 410 Gone
    await db.execute("UPDATE share_links SET expires_at = NOW() - INTERVAL '5 minutes' WHERE id = :i", {"i": sid1})
    assert (await client.get(f"/v1/share/{sid1}", headers=admin_headers)).status_code == 410
    assert (await client.get(f"/share/s/{sid1}")).status_code == 410
    assert (await client.get(f"/share/card/{sid1}.png")).status_code == 410
    assert (await client.get(f"/share/cover/{sid1}")).status_code == 410
    assert (await client.get(f"/share/go/{sid1}/apple", follow_redirects=False)).status_code == 410

    # Creating a share for the expired entity rotates to a fresh id
    r_new = await client.post("/v1/share", json={"kind": "track", "id": str(ids["track_id"])}, headers=admin_headers)
    assert r_new.status_code == 200
    assert r_new.json()["id"] != sid1

    # Re-expire and verify GC deletes expired share_links rows
    sid2 = r_new.json()["id"]
    await db.execute("UPDATE share_links SET expires_at = NOW() - INTERVAL '1 minute' WHERE id = :i", {"i": sid2})
    stats2 = await gc_service.run_gc()
    assert stats2["shares_deleted"] >= 1
    assert await db.fetch_one("SELECT 1 FROM share_links WHERE id = :i", {"i": sid2}) is None


def test_open_redirect_allowlist_and_artist_monet_card():
    assert share_service.is_safe_redirect_url("apple", "https://music.apple.com/song/123")
    assert not share_service.is_safe_redirect_url("apple", "http://music.apple.com/song/123")
    assert not share_service.is_safe_redirect_url("apple", "https://evil.example/song/123")
    assert not share_service.is_safe_redirect_url("yandex", "javascript:alert(1)")

    # Grayscale cover fallback still yields accessible high-contrast Monet CSS tokens
    seed = share_card._extract_seed(None, "Молчат Дома")
    tokens = share_card.build_monet_tokens(seed)
    assert share_card._contrast_ratio(tokens["dark"]["primary"], tokens["dark"]["on_primary"]) >= 4.5
    assert share_card._contrast_ratio(tokens["light"]["primary"], tokens["light"]["on_primary"]) >= 4.5
    assert "--md-sys-color-secondary-container" in tokens["dark"]["css"]
    assert "--md-sys-color-on-secondary-container" in tokens["light"]["css"]
    assert share_card._load_kwhale_logo_badge(38) is not None


@pytest.mark.asyncio
async def test_soundcloud_exact_and_fallback_resolver():
    entity = share_service.ShareEntity(
        kind="track",
        entity_id=77,
        title="Судно",
        subtitle="Молчат Дома",
        meta="Этажи · 2:21",
        duration_sec=141,
        image_id="placeholder",
        sources={"soundcloud": "994368454"},
    )

    async def ok_sc(_kind, _eid):
        return "https://soundcloud.com/molchat-doma/sudno"

    exact = await share_service.resolve_platform_links(entity, {"soundcloud"}, sc_resolver=ok_sc)
    assert len(exact) == 1
    assert exact[0].exact is True
    assert exact[0].url == "https://soundcloud.com/molchat-doma/sudno"

    async def fail_sc(_kind, _eid):
        return None

    fb = await share_service.resolve_platform_links(entity, {"soundcloud"}, sc_resolver=fail_sc)
    assert len(fb) == 1
    assert fb[0].exact is False
    assert fb[0].url.startswith("https://soundcloud.com/search/sounds?q=")
