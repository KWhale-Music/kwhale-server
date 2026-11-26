"""GC: ephemeral tracks TTL + unlike-grace file removal.

Runs in one daily cycle (same scheduler as ICM keepalive):
  1. ephemeral: tracks with no favorites, no local file, last_seen older
     than EPHEMERAL_TTL_DAYS -> DELETE (cascades track_sources)
     Also pinned: tracks of liked albums (library cards stay playable)
  2. unlike-grace: local sources with pending_delete_since older than
     UNLIKE_GRACE_DAYS and zero favorites -> delete file from disk + row
  3. search_cache: expired rows

Liked albums/artists (favorite_albums / favorite_artists) and the entities
reachable from liked tracks are pinned from the entity sweeps too.
"""

from __future__ import annotations

import logging
from pathlib import Path

from .. import db
from ..config import settings

log = logging.getLogger("kwhale.gc")

# a track belongs to a liked album (normalized album title + element-wise artist)
TRACK_IN_LIKED_ALBUM_SQL = """
    EXISTS (
        SELECT 1 FROM favorite_albums fa
        JOIN albums a ON a.id = fa.album_id
        JOIN artists ar ON ar.id = a.artist_id
        WHERE t.album IS NOT NULL
          AND lower(regexp_replace(t.album, '\\s+', '', 'g'))
              = lower(regexp_replace(a.title, '\\s+', '', 'g'))
          AND EXISTS (
              SELECT 1 FROM unnest(t.artists) AS an
              WHERE artist_matches(an, ar.name)
          )
    )
"""

# an artist name participates in a liked track (element-wise)
ARTIST_IN_LIKED_TRACK_SQL = """
    EXISTS (
        SELECT 1 FROM favorites f
        JOIN tracks t ON t.id = f.track_id
        WHERE EXISTS (
            SELECT 1 FROM unnest(t.artists) AS an
            WHERE artist_matches(an, ar.name)
        )
    )
"""

# a track belongs to a liked artist (its persisted discography is pinned too)
TRACK_IN_LIKED_ARTIST_SQL = """
    EXISTS (
        SELECT 1 FROM favorite_artists far
        JOIN artists ar ON ar.id = far.artist_id
        WHERE EXISTS (
            SELECT 1 FROM unnest(t.artists) AS an
            WHERE artist_matches(an, ar.name)
        )
    )
"""


async def purge_search_cache() -> int:
    """Drop expired connector responses; safe to call far more often than GC."""
    rows = await db.execute_returning(
        "DELETE FROM search_cache WHERE expires_at < NOW() RETURNING cache_key",
    )
    return len(rows)


async def run_gc() -> dict[str, int]:
    stats = {
        "tracks_deleted": 0,
        "files_deleted": 0,
        "cache_deleted": 0,
        "images_deleted": 0,
    }

    # 1. unlike-grace: files first (rows reference them)
    expired_files = await db.fetch_all(
        """
        SELECT s.id, s.file_path
        FROM track_sources s
        WHERE s.connector = 'local'
          AND s.permanent = FALSE
          AND s.pending_delete_since IS NOT NULL
          AND s.pending_delete_since < NOW() - make_interval(days => :grace)
          AND NOT EXISTS (
              SELECT 1 FROM favorites f WHERE f.track_id = s.track_id
          )
          AND NOT EXISTS (
              SELECT 1 FROM playlist_tracks pt WHERE pt.track_id = s.track_id
          )
          AND NOT EXISTS (
              SELECT 1 FROM embed_widget_tracks ewt
              JOIN embed_widgets ew ON ew.id = ewt.widget_id
              WHERE ew.active AND ewt.track_id = s.track_id
          )
        """,
        {"grace": settings.unlike_grace_days},
    )
    for row in expired_files:
        if row["file_path"]:
            p = Path(row["file_path"])
            if p.is_file():
                try:
                    p.unlink()
                    stats["files_deleted"] += 1
                except OSError as e:
                    log.warning("file delete %s: %s", p, e)
        await db.execute("DELETE FROM track_sources WHERE id = :i", {"i": row["id"]})

    # 2. ephemeral tracks (favorites, liked albums, playlists, and active embeds pin their tracks)
    deleted = await db.execute_returning(
        f"""
        DELETE FROM tracks t
        WHERE t.last_seen_at < NOW() - make_interval(days => :ttl)
          AND NOT EXISTS (SELECT 1 FROM favorites f WHERE f.track_id = t.id)
          AND NOT EXISTS (SELECT 1 FROM playlist_tracks pt WHERE pt.track_id = t.id)
          AND NOT EXISTS (
              SELECT 1 FROM embed_widget_tracks ewt
              JOIN embed_widgets ew ON ew.id = ewt.widget_id
              WHERE ew.active AND ewt.track_id = t.id
          )
          AND NOT EXISTS (SELECT 1 FROM download_jobs dj
                          WHERE dj.track_id = t.id AND dj.status IN ('pending', 'running'))
          AND NOT EXISTS (SELECT 1 FROM track_sources s
                          WHERE s.track_id = t.id AND s.connector = 'local')
          AND NOT {TRACK_IN_LIKED_ALBUM_SQL}
          AND NOT {TRACK_IN_LIKED_ARTIST_SQL}
        RETURNING t.id
        """,
        {"ttl": settings.ephemeral_ttl_days},
    )
    stats["tracks_deleted"] = len(deleted)

    # 3. ephemeral albums: keep if liked, if the artist is liked, pinned by an
    #    active embed widget, if a local track matches, or if a liked track's album matches
    deleted_albums = await db.execute_returning(
        """
        DELETE FROM albums a
        WHERE a.last_seen_at < NOW() - make_interval(days => :ttl)
          AND NOT EXISTS (SELECT 1 FROM favorite_albums fa WHERE fa.album_id = a.id)
          AND NOT EXISTS (SELECT 1 FROM favorite_artists far WHERE far.artist_id = a.artist_id)
          AND NOT EXISTS (
              SELECT 1 FROM embed_widgets ew
              WHERE ew.active
                AND ((ew.kind = 'album' AND ew.entity_id = a.id)
                     OR (ew.kind = 'artist' AND ew.entity_id = a.artist_id))
          )
          AND NOT EXISTS (
              SELECT 1 FROM tracks t
              JOIN track_sources s ON s.track_id = t.id AND s.connector = 'local'
              WHERE lower(regexp_replace(t.album, '\\s+', '', 'g'))
                  = lower(regexp_replace(a.title, '\\s+', '', 'g'))
          )
          AND NOT EXISTS (
              SELECT 1 FROM favorites f
              JOIN tracks t ON t.id = f.track_id
              WHERE t.album IS NOT NULL
                AND lower(regexp_replace(t.album, '\\s+', '', 'g'))
                    = lower(regexp_replace(a.title, '\\s+', '', 'g'))
                AND EXISTS (
                    SELECT 1 FROM unnest(t.artists) AS an
                    JOIN artists ar2 ON ar2.id = a.artist_id
                    WHERE artist_matches(an, ar2.name)
                )
          )
        RETURNING a.id
        """,
        {"ttl": settings.ephemeral_ttl_days},
    )
    stats["albums_deleted"] = len(deleted_albums)

    # 4. ephemeral artists (no albums left after the album sweep; liked artists,
    #    active artist embeds, and artists of liked tracks are pinned)
    deleted_artists = await db.execute_returning(
        f"""
        DELETE FROM artists ar
        WHERE ar.last_seen_at < NOW() - make_interval(days => :ttl)
          AND NOT EXISTS (SELECT 1 FROM albums al WHERE al.artist_id = ar.id)
          AND NOT EXISTS (SELECT 1 FROM favorite_artists far WHERE far.artist_id = ar.id)
          AND NOT EXISTS (
              SELECT 1 FROM embed_widgets ew
              WHERE ew.active AND ew.kind = 'artist' AND ew.entity_id = ar.id
          )
          AND NOT {ARTIST_IN_LIKED_TRACK_SQL}
        RETURNING ar.id
        """,
        {"ttl": settings.ephemeral_ttl_days},
    )
    stats["artists_deleted"] = len(deleted_artists)

    # 5. search cache
    stats["cache_deleted"] = await purge_search_cache()

    # 6. disposable covers. A cover is pinned while it belongs to a liked
    # track, a liked album/artist, a local/downloaded track, a queued
    # download, or an active embed widget. Everything else expires after the TTL.
    pinned_urls = """
        SELECT t.cover_url AS url
        FROM favorites f JOIN tracks t ON t.id = f.track_id
        WHERE COALESCE(t.cover_url, '') <> ''
        UNION
        SELECT m.cover_url AS url
        FROM favorites f JOIN track_source_metadata m ON m.track_id = f.track_id
        WHERE COALESCE(m.cover_url, '') <> ''
        UNION
        SELECT t.cover_url AS url
        FROM tracks t JOIN track_sources s
          ON s.track_id = t.id AND s.connector = 'local'
        WHERE COALESCE(t.cover_url, '') <> ''
        UNION
        SELECT a.cover_url AS url
        FROM favorite_albums fa JOIN albums a ON a.id = fa.album_id
        WHERE COALESCE(a.cover_url, '') <> ''
        UNION
        SELECT ar.cover_url AS url
        FROM favorite_artists far JOIN artists ar ON ar.id = far.artist_id
        WHERE COALESCE(ar.cover_url, '') <> ''
        UNION
        SELECT t.cover_url AS url
        FROM download_jobs dj JOIN tracks t ON t.id = dj.track_id
        WHERE COALESCE(t.cover_url, '') <> ''
        UNION
        SELECT t.cover_url AS url
        FROM embed_widget_tracks ewt
        JOIN embed_widgets ew ON ew.id = ewt.widget_id
        JOIN tracks t ON t.id = ewt.track_id
        WHERE ew.active AND COALESCE(t.cover_url, '') <> ''
    """
    deleted_images = await db.execute_returning(
        f"""
        WITH pinned AS ({pinned_urls})
        DELETE FROM images i
        WHERE i.id <> 'placeholder'
          AND COALESCE(i.last_accessed_at, i.fetched_at, i.created_at)
              < NOW() - make_interval(days => :ttl)
          AND NOT EXISTS (SELECT 1 FROM pinned p WHERE p.url = i.source_url)
          AND NOT EXISTS (SELECT 1 FROM playlists pl WHERE pl.cover_image_id = i.id)
          AND NOT EXISTS (
              SELECT 1 FROM share_links sl
              WHERE sl.image_id = i.id AND sl.expires_at > NOW()
          )
          AND NOT EXISTS (
              SELECT 1 FROM embed_widgets ew
              WHERE ew.active AND ew.image_id = i.id
          )
        RETURNING i.id
        """,
        {"ttl": settings.image_cache_ttl_days},
    )
    stats["images_deleted"] = len(deleted_images)

    # Stale artist sync states (one row per artist) are tiny, but keep the
    # table bounded: non-liked states older than a month can be dropped.
    await db.execute(
        """
        DELETE FROM artist_sync_state s
        WHERE s.expires_at IS NOT NULL
          AND s.expires_at < NOW() - INTERVAL '30 days'
          AND NOT EXISTS (
              SELECT 1 FROM favorite_artists fa WHERE fa.artist_id = s.artist_id
          )
        """
    )

    # 7. raw events retention (§10: default 180 days), idempotency keys & expired refresh tokens
    deleted_events = await db.execute_returning(
        """
        DELETE FROM events
        WHERE COALESCE(occurred_at, received_at) < NOW() - make_interval(days => :days)
        RETURNING id
        """,
        {"days": settings.events_retention_days},
    )
    stats["events_deleted"] = len(deleted_events)

    await db.execute(
        "DELETE FROM idempotency_keys WHERE created_at < NOW() - INTERVAL '7 days'"
    )
    await db.execute(
        "DELETE FROM refresh_tokens WHERE expires_at < NOW() OR (revoked_at IS NOT NULL AND revoked_at < NOW() - INTERVAL '7 days')"
    )
    deleted_shares = await db.execute_returning(
        "DELETE FROM share_links WHERE expires_at <= NOW() RETURNING id"
    )
    stats["shares_deleted"] = len(deleted_shares)

    # 8. LRU stream cache eviction
    from .stream_service import evict_stream_cache

    sc_stats = evict_stream_cache()
    stats["stream_cache_evicted"] = sc_stats.get("evicted_files", 0)

    if any(stats.values()):
        log.info("gc done: %s", stats)
    return stats


async def purge_incoming() -> dict[str, int]:
    """Remove staged files older than 24h (crash leftovers)."""

    from .library import staging_dir

    root = staging_dir()
    stats = {"incoming_purged": 0}
    cutoff = 24 * 3600
    import time

    now = time.time()
    for p in root.rglob("*"):
        if p.is_file() and now - p.stat().st_mtime > cutoff:
            try:
                p.unlink()
                stats["incoming_purged"] += 1
            except OSError as e:
                log.warning("incoming purge %s: %s", p, e)
    return stats
