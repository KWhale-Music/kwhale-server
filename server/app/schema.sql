-- KWhale-v3 schema
-- NB: avoid semicolons in comments (naive statement splitter)

CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE EXTENSION IF NOT EXISTS unaccent;

-- Keep SQL artist matching identical to services.normalize.normalize_artist
-- (dedup_artist_key with spaces removed): Latin/Cyrillic homoglyph folding,
-- NFKD-ish diacritics, Cyrillic ё/й folding, punctuation removed.
-- Order matters: the homoglyph translate runs BEFORE unaccent so «ё» folds to
-- Latin "e" exactly like Python's _HOMOGLYPHS table. Stock unaccent() folds «ё»
-- to Cyrillic "е" instead and knows nothing about т→t / н→h look-alikes. The
-- artist-page prefilter then matched nothing for Cyrillic names (the
-- «Тёмный Принц» bug, 2026-11-18). The translate table below is the _HOMOGLYPHS
-- map from services/normalize.py — keep the two in sync.
CREATE OR REPLACE FUNCTION artist_norm(text) RETURNS text
LANGUAGE sql STABLE PARALLEL SAFE AS $$
    SELECT lower(regexp_replace(translate(public.unaccent(translate($1,
        'АВЕКМНОРСТУХІЈЅавекмнорстухіјѕёЁ',
        'ABEKMHOPCTYXIJSabekmhopctyxijseE')), 'йЙ', 'иИ'), '[^\w]', '', 'g'))
$$;

-- Artist strings from SoundCloud/ICM sometimes contain "A & B" in one field.
-- Matching against either the whole string or one of its tokens keeps tracks
-- findable without rewriting group names such as "MONSTA X" or "AC/DC".
CREATE OR REPLACE FUNCTION artist_tokens(text) RETURNS text[]
LANGUAGE sql STABLE PARALLEL SAFE AS $$
    SELECT array_remove(
        ARRAY(
            SELECT public.artist_norm(part)
            FROM regexp_split_to_table(
                $1,
                '\s*(?:,|&|\bvs\b\.?|\bfeaturing\b|\bfeat\b\.?|\bft\b\.?)\s*',
                'i'
            ) AS part
        ),
        ''
    )
$$;

CREATE OR REPLACE FUNCTION artist_matches(track_artist text, wanted text) RETURNS boolean
LANGUAGE sql STABLE PARALLEL SAFE AS $$
    SELECT public.artist_norm(track_artist) = public.artist_norm(wanted)
        OR public.artist_norm(wanted) = ANY(public.artist_tokens(track_artist))
$$;

-- IMMUTABLE wrapper so tracks.artists (text[]) can back a generated column
-- and a trigram index. array_to_string itself is only STABLE, which PostgreSQL
-- refuses in generated columns and expression indexes.
CREATE OR REPLACE FUNCTION artists_text_imm(text[]) RETURNS text
LANGUAGE sql IMMUTABLE PARALLEL SAFE AS $$
    SELECT array_to_string($1, ' ')
$$;

-- Same trick for artist name lookups: artist_norm() is STABLE (unaccent), so
-- it cannot back an expression index. Every search that materialises a new
-- artist used to full-scan the 56k-row artists table, and this index makes the
-- fallback lookup a point read. Body must stay byte-identical to artist_norm()
-- or the generated column drifts from query-time normalization.
CREATE OR REPLACE FUNCTION artist_norm_imm(text) RETURNS text
LANGUAGE sql IMMUTABLE PARALLEL SAFE AS $$
    SELECT lower(regexp_replace(translate(public.unaccent(translate($1,
        'АВЕКМНОРСТУХІЈЅавекмнорстухіјѕёЁ',
        'ABEKMHOPCTYXIJSabekmhopctyxijseE')), 'йЙ', 'иИ'), '[^\w]', '', 'g'))
$$;
-- One Track entity, many sources. Local file = source with connector='local'.

CREATE TABLE IF NOT EXISTS users (
    id            SERIAL PRIMARY KEY,
    username      TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS tracks (
    id            BIGSERIAL PRIMARY KEY,
    title         TEXT NOT NULL,
    artists       TEXT[] NOT NULL DEFAULT '{}',
    album         TEXT,
    duration_sec  INT,                -- canonical (best-source) duration
    isrc          TEXT,
    cover_url     TEXT,
    year          INT,
    track_number  INT,
    track_total   INT,
    disc_number   INT,
    disc_total    INT,
    explicit      BOOLEAN,
    genre         TEXT,
    genres_norm   TEXT[] NOT NULL DEFAULT '{}',  -- canonical families (services.genres)
    identity_hash TEXT,               -- services.identity: (title, primary artist) key
    last_seen_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE UNIQUE INDEX IF NOT EXISTS tracks_isrc_key
    ON tracks (isrc) WHERE isrc IS NOT NULL;

-- NB: the tracks_genres_norm index is created by db.apply_schema migrations
-- (AFTER the ALTER that adds genres_norm to pre-existing tables) — adding it
-- here would fail on every deployed database where CREATE TABLE is a no-op.
-- Same rule as tracks_identity_hash below.

-- NB: the tracks_identity_hash index is created by db.apply_schema migrations
-- (AFTER the ALTER that adds identity_hash to pre-existing tables) — adding it
-- here would fail on every deployed database where CREATE TABLE is a no-op.

-- Materialised "Artist A Artist B" text for trigram search. _fuzzy_existing()
-- used to scan all 40k tracks for every candidate (~50 ms x 180 = 9 s per
-- search) and the GIN index below turns that into a few milliseconds.
ALTER TABLE tracks ADD COLUMN IF NOT EXISTS artists_text TEXT
    GENERATED ALWAYS AS (public.artists_text_imm(artists)) STORED;
CREATE INDEX IF NOT EXISTS tracks_artists_text_trgm
    ON tracks USING gin (artists_text gin_trgm_ops);
CREATE INDEX IF NOT EXISTS tracks_title_trgm
    ON tracks USING gin (title gin_trgm_ops);

-- Normalised artist text for the artist-page fuzzy lookup. artist_matches()
-- re-normalised every row (56k scans, ~800 ms per artist open). The trigram
-- index narrows candidates and the EXISTS check keeps token semantics.
ALTER TABLE tracks ADD COLUMN IF NOT EXISTS artists_norm TEXT
    GENERATED ALWAYS AS (public.artist_norm_imm(public.artists_text_imm(artists))) STORED;
CREATE INDEX IF NOT EXISTS tracks_artists_norm_trgm
    ON tracks USING gin (artists_norm gin_trgm_ops);

CREATE TABLE IF NOT EXISTS track_sources (
    id                   BIGSERIAL PRIMARY KEY,
    track_id             BIGINT NOT NULL REFERENCES tracks(id) ON DELETE CASCADE,
    connector            TEXT NOT NULL,           -- 'local' | 'icm' | 'yandex' | 'soundcloud'
    external_id          TEXT NOT NULL,          -- provider id or filename for local
    quality_tier         INT NOT NULL DEFAULT 0, -- higher = better
    mime                 TEXT,
    file_path            TEXT,                   -- set for local sources
    available            BOOLEAN NOT NULL DEFAULT TRUE,
    pending_delete_since TIMESTAMPTZ,            -- unlike-grace marker (local only)
    updated_at           TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (connector, external_id)
);

CREATE INDEX IF NOT EXISTS track_sources_track ON track_sources (track_id);
CREATE INDEX IF NOT EXISTS track_sources_local ON track_sources (track_id) WHERE connector = 'local';

CREATE TABLE IF NOT EXISTS favorites (
    user_id    INT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    track_id   BIGINT NOT NULL REFERENCES tracks(id) ON DELETE CASCADE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (user_id, track_id)
);
-- The PK starts with user_id, so EXISTS (... WHERE track_id=...) needs its
-- own index or it seq-scans the whole table per candidate row.
CREATE INDEX IF NOT EXISTS favorites_track ON favorites (track_id);

CREATE TABLE IF NOT EXISTS download_jobs (
    id         BIGSERIAL PRIMARY KEY,
    track_id   BIGINT NOT NULL UNIQUE REFERENCES tracks(id) ON DELETE CASCADE,
    status     TEXT NOT NULL DEFAULT 'pending',  -- pending|running|done|failed
    error      TEXT,
    attempts   INT NOT NULL DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ── Audio quality of the local file (measured, not source tier) ──────────────
-- One row per track: the currently playable local copy. local_source_id is the
-- track_sources row the measurement belongs to. Deleting that file row removes
-- the quality row too. Tier: 0 unknown, 1 <128, 2 128-191, 3 192-255,
-- 4 256-319, 5 320+, 6 lossless.
CREATE TABLE IF NOT EXISTS track_quality (
    track_id        BIGINT PRIMARY KEY REFERENCES tracks(id) ON DELETE CASCADE,
    local_source_id BIGINT REFERENCES track_sources(id) ON DELETE CASCADE,
    file_path       TEXT NOT NULL,
    container       TEXT,
    codec           TEXT,
    bitrate_kbps    INT,
    sample_rate     INT,
    bit_depth       INT,
    channels        INT,
    lossless        BOOLEAN NOT NULL DEFAULT FALSE,
    tier            SMALLINT NOT NULL DEFAULT 0,
    bytes           BIGINT,
    duration_sec    REAL,
    measured_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    error           TEXT
);
CREATE INDEX IF NOT EXISTS track_quality_source ON track_quality (local_source_id);
CREATE INDEX IF NOT EXISTS track_quality_tier ON track_quality (tier);
CREATE INDEX IF NOT EXISTS track_quality_bitrate ON track_quality (bitrate_kbps);
-- force re-download: replace the existing local copy after a verified download
ALTER TABLE download_jobs ADD COLUMN IF NOT EXISTS replace_local BOOLEAN NOT NULL DEFAULT FALSE;

-- TTL cache of raw connector search responses (latency shield for ICM etc.)
CREATE TABLE IF NOT EXISTS search_cache (
    cache_key   TEXT PRIMARY KEY,
    connector   TEXT NOT NULL,
    result_json JSONB NOT NULL,
    expires_at  TIMESTAMPTZ NOT NULL
);
CREATE INDEX IF NOT EXISTS search_cache_expires ON search_cache (expires_at);

-- ── Entity graph: artists / albums (from connector searches) ─────────────────
-- Same pattern as tracks: stable DB ids, sources[] as fallback, ephemeral GC
-- Local tracks keep their text album fields (albums table is the browse entity)

CREATE TABLE IF NOT EXISTS artists (
    id           BIGSERIAL PRIMARY KEY,
    name         TEXT NOT NULL,
    cover_url    TEXT,
    genre        TEXT,
    last_seen_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS artists_norm_imm_idx ON artists (public.artist_norm_imm(name));

-- Explicit track -> canonical artist links (avoids the expensive fuzzy
-- artist_matches() join in /v1/library/artists).
CREATE TABLE IF NOT EXISTS track_artists (
    track_id  BIGINT NOT NULL REFERENCES tracks(id) ON DELETE CASCADE,
    artist_id BIGINT NOT NULL REFERENCES artists(id) ON DELETE CASCADE,
    PRIMARY KEY (track_id, artist_id)
);
CREATE INDEX IF NOT EXISTS track_artists_artist ON track_artists (artist_id, track_id);

CREATE TABLE IF NOT EXISTS artist_sources (
    id          BIGSERIAL PRIMARY KEY,
    artist_id   BIGINT NOT NULL REFERENCES artists(id) ON DELETE CASCADE,
    connector   TEXT NOT NULL,
    external_id TEXT NOT NULL,
    available   BOOLEAN NOT NULL DEFAULT TRUE,
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (connector, external_id)
);

CREATE TABLE IF NOT EXISTS artist_sync_state (
    artist_id   BIGINT PRIMARY KEY REFERENCES artists(id) ON DELETE CASCADE,
    status      TEXT NOT NULL DEFAULT 'idle',
    started_at  TIMESTAMPTZ,
    finished_at TIMESTAMPTZ,
    expires_at  TIMESTAMPTZ,
    track_count INT NOT NULL DEFAULT 0,
    album_count INT NOT NULL DEFAULT 0,
    error       TEXT
);
CREATE INDEX IF NOT EXISTS artist_sync_state_expires ON artist_sync_state (expires_at);

CREATE TABLE IF NOT EXISTS albums (
    id           BIGSERIAL PRIMARY KEY,
    title        TEXT NOT NULL,
    artist_id    BIGINT NOT NULL REFERENCES artists(id) ON DELETE CASCADE,
    year         INT,
    release_date DATE,
    cover_url    TEXT,
    track_count  INT,
    last_seen_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
ALTER TABLE albums ADD COLUMN IF NOT EXISTS release_date DATE;
CREATE INDEX IF NOT EXISTS albums_release_date ON albums (release_date);

CREATE TABLE IF NOT EXISTS album_sources (
    id          BIGSERIAL PRIMARY KEY,
    album_id    BIGINT NOT NULL REFERENCES albums(id) ON DELETE CASCADE,
    connector   TEXT NOT NULL,
    external_id TEXT NOT NULL,
    available   BOOLEAN NOT NULL DEFAULT TRUE,
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (connector, external_id)
);

CREATE INDEX IF NOT EXISTS albums_artist ON albums (artist_id);
CREATE INDEX IF NOT EXISTS albums_title_trgm ON albums USING gin (title gin_trgm_ops);

-- ── Entity favorites: liked albums / artists (explicit library saves) ────────
-- Liking a track downloads a file and saves the track. Liking an album or an
-- artist only saves the card to the library (no download side effects).

CREATE TABLE IF NOT EXISTS favorite_albums (
    user_id    INT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    album_id   BIGINT NOT NULL REFERENCES albums(id) ON DELETE CASCADE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (user_id, album_id)
);

CREATE TABLE IF NOT EXISTS favorite_artists (
    user_id    INT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    artist_id  BIGINT NOT NULL REFERENCES artists(id) ON DELETE CASCADE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (user_id, artist_id)
);

-- ── Playlists (user + future recommendation system playlists) ────────────────

CREATE TABLE IF NOT EXISTS playlists (
    id          BIGSERIAL PRIMARY KEY,
    user_id     INT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    name        TEXT NOT NULL,
    description TEXT,
    is_system   BOOLEAN NOT NULL DEFAULT FALSE,
    source      TEXT NOT NULL DEFAULT 'user',  -- 'user' | 'recommendation'
    cover_image_id TEXT,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
ALTER TABLE playlists ADD COLUMN IF NOT EXISTS cover_image_id TEXT;
CREATE INDEX IF NOT EXISTS playlists_user ON playlists (user_id);

CREATE TABLE IF NOT EXISTS weekly_release_playlists (
    user_id     INT PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
    playlist_id BIGINT REFERENCES playlists(id) ON DELETE SET NULL,
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS playlist_tracks (
    playlist_id BIGINT NOT NULL REFERENCES playlists(id) ON DELETE CASCADE,
    track_id    BIGINT NOT NULL REFERENCES tracks(id) ON DELETE CASCADE,
    position    INT NOT NULL,
    added_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (playlist_id, track_id)
);
CREATE INDEX IF NOT EXISTS playlist_tracks_pos ON playlist_tracks (playlist_id, position);
-- ── Dislikes (per-user, no download side effects) ────────────────────────────

CREATE TABLE IF NOT EXISTS dislikes (
    user_id    INT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    track_id   BIGINT NOT NULL REFERENCES tracks(id) ON DELETE CASCADE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (user_id, track_id)
);

-- ── Playlist visibility + follows ────────────────────────────────────────────

ALTER TABLE playlists ADD COLUMN IF NOT EXISTS visibility TEXT NOT NULL DEFAULT 'private';

CREATE TABLE IF NOT EXISTS playlist_follows (
    user_id     INT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    playlist_id BIGINT NOT NULL REFERENCES playlists(id) ON DELETE CASCADE,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (user_id, playlist_id)
);

-- ── Download job targets (bulk download: track/album/playlist) ───────────────

ALTER TABLE download_jobs ADD COLUMN IF NOT EXISTS target_type TEXT;
ALTER TABLE download_jobs ADD COLUMN IF NOT EXISTS target_id BIGINT;

-- ── Source accounts: which services the user connected (web-ui /admin) ───────
-- Secrets live AES-GCM encrypted in payload_enc, profile is non-secret display info

CREATE TABLE IF NOT EXISTS source_accounts (
    connector   TEXT PRIMARY KEY,
    kind        TEXT NOT NULL DEFAULT 'account',   -- 'account' | 'env'
    payload_enc TEXT,                              -- gcm1:<b64> envelope, NULL for env-kind
    profile     JSONB NOT NULL DEFAULT '{}',
    status      TEXT NOT NULL DEFAULT 'connected', -- connected|expired|error|disabled
    enabled     BOOLEAN NOT NULL DEFAULT TRUE,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ── KWhale v1: users, invites, tokens, metadata, images, events ─────────────────

ALTER TABLE users ADD COLUMN IF NOT EXISTS email TEXT;
ALTER TABLE users ADD COLUMN IF NOT EXISTS display_name TEXT;
ALTER TABLE users ADD COLUMN IF NOT EXISTS role TEXT NOT NULL DEFAULT 'user';
CREATE UNIQUE INDEX IF NOT EXISTS users_email_key ON users (lower(email)) WHERE email IS NOT NULL;

CREATE TABLE IF NOT EXISTS invites (
    id         BIGSERIAL PRIMARY KEY,
    token_hash TEXT UNIQUE NOT NULL,
    created_by INT REFERENCES users(id) ON DELETE SET NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    expires_at TIMESTAMPTZ NOT NULL,
    used_by    INT REFERENCES users(id) ON DELETE SET NULL,
    used_at    TIMESTAMPTZ,
    revoked_at TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS refresh_tokens (
    id         BIGSERIAL PRIMARY KEY,
    user_id    INT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    token_hash TEXT UNIQUE NOT NULL,
    device_id  TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    expires_at TIMESTAMPTZ NOT NULL,
    revoked_at TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS refresh_tokens_user ON refresh_tokens (user_id);

CREATE TABLE IF NOT EXISTS track_source_metadata (
    track_id    BIGINT NOT NULL REFERENCES tracks(id) ON DELETE CASCADE,
    connector   TEXT NOT NULL,
    external_id TEXT NOT NULL,
    title       TEXT,
    artists     TEXT[],
    album       TEXT,
    year        INT,
    track_number INT,
    track_total INT,
    disc_number INT,
    disc_total  INT,
    genre       TEXT,
    cover_url   TEXT,
    isrc        TEXT,
    duration_sec INT,
    quality_tier INT,
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (track_id, connector, external_id)
);

CREATE TABLE IF NOT EXISTS source_metadata_priority (
    connector TEXT PRIMARY KEY,
    priority  INT NOT NULL
);
INSERT INTO source_metadata_priority (connector, priority) VALUES
    ('icm', 10), ('yandex', 20), ('qobuz', 30), ('tidal', 40),
    ('deezer', 50), ('zvuk', 60), ('vk', 70), ('ytmusic', 80), ('soundcloud', 90)
ON CONFLICT (connector) DO NOTHING;

CREATE TABLE IF NOT EXISTS images (
    id         TEXT PRIMARY KEY,
    source_url TEXT NOT NULL,
    mime       TEXT,
    data       BYTEA,
    width      INT,
    height     INT,
    etag       TEXT,
    fetched_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_accessed_at TIMESTAMPTZ
);
ALTER TABLE images ADD COLUMN IF NOT EXISTS created_at TIMESTAMPTZ NOT NULL DEFAULT NOW();
ALTER TABLE images ADD COLUMN IF NOT EXISTS last_accessed_at TIMESTAMPTZ;
CREATE INDEX IF NOT EXISTS images_ttl ON images (last_accessed_at, created_at);

CREATE TABLE IF NOT EXISTS image_variants (
    image_id TEXT NOT NULL REFERENCES images(id) ON DELETE CASCADE,
    size     TEXT NOT NULL,
    mime     TEXT NOT NULL,
    data     BYTEA NOT NULL,
    etag     TEXT NOT NULL,
    PRIMARY KEY (image_id, size)
);

CREATE TABLE IF NOT EXISTS events (
    id           UUID PRIMARY KEY,
    user_id      INT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    type         TEXT NOT NULL,
    track_id     BIGINT,
    playback_id  TEXT,
    payload      JSONB NOT NULL DEFAULT '{}',
    occurred_at  TIMESTAMPTZ,
    received_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS events_user_time ON events (user_id, occurred_at DESC);
CREATE INDEX IF NOT EXISTS events_track ON events (track_id);

CREATE TABLE IF NOT EXISTS search_history (
    id         BIGSERIAL PRIMARY KEY,
    user_id    INT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    query      TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS search_history_user ON search_history (user_id, created_at DESC);

CREATE TABLE IF NOT EXISTS idempotency_keys (
    user_id    INT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    endpoint   TEXT NOT NULL,
    key        TEXT NOT NULL,
    response   JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (user_id, endpoint, key)
);

-- User flagging a track for admin censorship review (client: "Отправить на
-- цензуру"). One open report per (user, track) -- DELETE cancels it.
CREATE TABLE IF NOT EXISTS censorship_reports (
    id          BIGSERIAL PRIMARY KEY,
    user_id     INT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    track_id    BIGINT NOT NULL REFERENCES tracks(id) ON DELETE CASCADE,
    reason      TEXT NOT NULL DEFAULT 'censored',
    comment     TEXT,
    position_ms BIGINT,
    status      TEXT NOT NULL DEFAULT 'pending',
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (user_id, track_id)
);
CREATE INDEX IF NOT EXISTS censorship_reports_status ON censorship_reports (status, created_at DESC);

-- App releases for the in-app updater. APK bytes live on disk
-- (data/app-releases/{id}.apk), the row holds version + metadata.
CREATE TABLE IF NOT EXISTS app_releases (
    id           BIGSERIAL PRIMARY KEY,
    version_code INT NOT NULL,
    version_name TEXT NOT NULL,
    notes        TEXT,
    file_path    TEXT NOT NULL,
    sha256       TEXT,
    size_bytes   BIGINT,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (version_code)
);

CREATE TABLE IF NOT EXISTS service_status (
    name       TEXT PRIMARY KEY,
    status     TEXT NOT NULL,
    message    TEXT,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

ALTER TABLE track_sources ADD COLUMN IF NOT EXISTS permanent BOOLEAN NOT NULL DEFAULT FALSE;
ALTER TABLE tracks ADD COLUMN IF NOT EXISTS has_lyrics BOOLEAN NOT NULL DEFAULT FALSE;
ALTER TABLE albums ADD COLUMN IF NOT EXISTS explicit BOOLEAN NOT NULL DEFAULT FALSE;
ALTER TABLE albums ADD COLUMN IF NOT EXISTS album_type TEXT;

CREATE TABLE IF NOT EXISTS user_onboarding (
    user_id    INT PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
    genres     TEXT[] NOT NULL DEFAULT '{}',
    artists    TEXT[] NOT NULL DEFAULT '{}',
    artist_ids TEXT[] NOT NULL DEFAULT '{}',
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
ALTER TABLE user_onboarding ADD COLUMN IF NOT EXISTS artist_ids TEXT[] NOT NULL DEFAULT '{}';

CREATE TABLE IF NOT EXISTS proxy_settings (
    id            INT PRIMARY KEY DEFAULT 1 CHECK (id = 1),
    enabled       BOOLEAN NOT NULL DEFAULT TRUE,
    default_proxy TEXT NOT NULL DEFAULT '',
    overrides     JSONB NOT NULL DEFAULT '{}',
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ── Read-only views for the recommendation engine (§10) ──────────────────────

CREATE OR REPLACE VIEW v_user_track_interactions AS
SELECT
    u.id AS user_id,
    t.id AS track_id,
    COALESCE(ev.play_count, 0) AS play_count,
    COALESCE(ev.complete_count, 0) AS complete_count,
    COALESCE(ev.skip_count, 0) AS skip_count,
    (f.track_id IS NOT NULL) AS is_liked,
    (d.track_id IS NOT NULL) AS is_disliked,
    GREATEST(ev.last_played_at, f.created_at, d.created_at) AS last_interaction_at
FROM users u
CROSS JOIN tracks t
LEFT JOIN (
    SELECT
        user_id,
        track_id,
        COUNT(*) FILTER (WHERE type = 'play') AS play_count,
        COUNT(*) FILTER (WHERE type = 'complete') AS complete_count,
        COUNT(*) FILTER (WHERE type = 'skip') AS skip_count,
        MAX(COALESCE(occurred_at, received_at)) FILTER (WHERE type = 'play') AS last_played_at
    FROM events
    WHERE track_id IS NOT NULL
    GROUP BY user_id, track_id
) ev ON ev.user_id = u.id AND ev.track_id = t.id
LEFT JOIN favorites f ON f.user_id = u.id AND f.track_id = t.id
LEFT JOIN dislikes d ON d.user_id = u.id AND d.track_id = t.id
WHERE ev.track_id IS NOT NULL OR f.track_id IS NOT NULL OR d.track_id IS NOT NULL;

CREATE OR REPLACE VIEW v_user_artist_affinities AS
SELECT
    i.user_id,
    an.artist_name,
    COUNT(*) FILTER (WHERE i.is_liked) AS liked_tracks,
    COUNT(*) FILTER (WHERE i.is_disliked) AS disliked_tracks,
    SUM(i.play_count) AS play_count,
    SUM(i.complete_count) AS complete_count,
    SUM(i.skip_count) AS skip_count,
    MAX(i.last_interaction_at) AS last_interaction_at
FROM v_user_track_interactions i
JOIN tracks t ON t.id = i.track_id
CROSS JOIN LATERAL unnest(t.artists) AS an(artist_name)
GROUP BY i.user_id, an.artist_name;

CREATE OR REPLACE VIEW v_user_onboarding AS
SELECT
    user_id,
    genres,
    artists,
    updated_at
FROM user_onboarding;

-- ── Recommendation layer (see docs/RECOMMENDATIONS.md) ───────────────────────
-- NB: pgvector image is required, extension must exist before vector columns

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS track_features (
    track_id           BIGINT PRIMARY KEY REFERENCES tracks(id) ON DELETE CASCADE,
    bpm                REAL,
    beat_confidence    REAL,
    key                INTEGER,
    mode               INTEGER,
    loudness_lufs      REAL,
    dynamic_complexity REAL,
    danceability       REAL,
    onset_rate         REAL,
    spectral_centroid  REAL,
    spectral_rolloff   REAL,
    spectral_flux      REAL,
    mfcc_mean          REAL[],
    mfcc_std           REAL[],
    energy             REAL,
    valence            REAL,
    voice_instrumental REAL,
    dsp_extra          JSONB NOT NULL DEFAULT '{}',
    tags               JSONB NOT NULL DEFAULT '{}',
    subgenres          TEXT[] NOT NULL DEFAULT '{}',
    moods              TEXT[] NOT NULL DEFAULT '{}',
    textures           TEXT[] NOT NULL DEFAULT '{}',
    scenes             TEXT[] NOT NULL DEFAULT '{}',
    language           TEXT,
    era                TEXT,
    vibe_summary       TEXT,
    semantic_energy    REAL,
    semantic_valence   REAL,
    acousticness       REAL,
    instrumentalness   REAL,
    tags_source        TEXT,
    tags_version       TEXT,
    tags_indexed_at    TIMESTAMPTZ,
    lyrics             TEXT,
    extractor_version  TEXT,
    indexed_at         TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS track_features_tags ON track_features USING gin (tags jsonb_path_ops);

CREATE TABLE IF NOT EXISTS rec_feature_stats (
    id         TEXT PRIMARY KEY,
    stats      JSONB NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS user_mood_profiles (
    user_id        INT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    preset         TEXT NOT NULL,
    audio_centroid REAL[],
    pos_count      INT NOT NULL DEFAULT 0,
    neg_count      INT NOT NULL DEFAULT 0,
    updated_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (user_id, preset)
);

CREATE TABLE IF NOT EXISTS track_emb_text (
    track_id      BIGINT PRIMARY KEY REFERENCES tracks(id) ON DELETE CASCADE,
    model_version TEXT NOT NULL,
    vec           vector(1024) NOT NULL,
    indexed_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS track_emb_text_hnsw ON track_emb_text USING hnsw (vec vector_cosine_ops);

CREATE TABLE IF NOT EXISTS track_emb_audio (
    track_id      BIGINT PRIMARY KEY REFERENCES tracks(id) ON DELETE CASCADE,
    model_version TEXT NOT NULL,
    vec           vector(512) NOT NULL,
    indexed_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS track_emb_audio_hnsw ON track_emb_audio USING hnsw (vec vector_cosine_ops);

CREATE TABLE IF NOT EXISTS index_jobs (
    id            BIGSERIAL PRIMARY KEY,
    track_id      BIGINT NOT NULL REFERENCES tracks(id) ON DELETE CASCADE,
    kind          TEXT NOT NULL,
    model_version TEXT NOT NULL,
    status        TEXT NOT NULL DEFAULT 'pending',
    attempts      INT NOT NULL DEFAULT 0,
    last_error    TEXT,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (track_id, kind, model_version)
);
CREATE INDEX IF NOT EXISTS index_jobs_pending ON index_jobs (status, id);

CREATE TABLE IF NOT EXISTS scoring_profiles (
    label      TEXT PRIMARY KEY,
    params     JSONB NOT NULL,
    version    INT NOT NULL DEFAULT 1,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS events_user_type_time ON events (user_id, type, occurred_at DESC);
CREATE INDEX IF NOT EXISTS events_mix_id_idx
    ON events ((payload->'context'->>'mixId'))
    WHERE payload->'context'->>'mixId' IS NOT NULL;

CREATE MATERIALIZED VIEW IF NOT EXISTS mv_user_track_stats AS
SELECT user_id,
       track_id,
       COUNT(*) FILTER (WHERE type = 'play')     AS plays,
       COUNT(*) FILTER (
           WHERE type = 'complete'
              OR (
                  type IN ('skip', 'pause')
                  AND (payload->>'listenedMs') ~ '^\d+$'
                  AND (payload->>'trackDurationMs') ~ '^\d+$'
                  AND (payload->>'trackDurationMs')::bigint > 0
                  AND (payload->>'listenedMs')::double precision
                      >= 0.85 * (payload->>'trackDurationMs')::double precision
              )
       ) AS completes,
       COUNT(*) FILTER (
           WHERE type = 'skip'
             AND NOT (
                 (payload->>'listenedMs') ~ '^\d+$'
                 AND (payload->>'trackDurationMs') ~ '^\d+$'
                 AND (payload->>'trackDurationMs')::bigint > 0
                 AND (payload->>'listenedMs')::double precision
                     >= 0.85 * (payload->>'trackDurationMs')::double precision
             )
       ) AS skips,
       COUNT(*) FILTER (WHERE type = 'repeat')   AS repeats,
       MAX(COALESCE(occurred_at, received_at)) FILTER (WHERE type IN ('play', 'complete', 'repeat')) AS last_played
FROM events
WHERE track_id IS NOT NULL
GROUP BY user_id, track_id;
CREATE UNIQUE INDEX IF NOT EXISTS mv_user_track_stats_pk ON mv_user_track_stats (user_id, track_id);

CREATE MATERIALIZED VIEW IF NOT EXISTS mv_user_artist_stats AS
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
GROUP BY s.user_id, an.artist_name;
CREATE UNIQUE INDEX IF NOT EXISTS mv_user_artist_stats_pk ON mv_user_artist_stats (user_id, artist_name);

CREATE MATERIALIZED VIEW IF NOT EXISTS mv_item_cooccurrence AS
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
            AND (sk.payload->>'listenedMs') ~ '^\d+$'
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
GROUP BY track_a, track_b;
CREATE UNIQUE INDEX IF NOT EXISTS mv_item_cooccurrence_pk ON mv_item_cooccurrence (track_a, track_b);

-- ── Metadata enrichment proposals for manual uploads (user confirms) ───────
CREATE TABLE IF NOT EXISTS track_enrichment_proposals (
    id          BIGSERIAL PRIMARY KEY,
    track_id    BIGINT NOT NULL REFERENCES tracks(id) ON DELETE CASCADE,
    user_id     INT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    proposed    JSONB NOT NULL DEFAULT '{}',      -- {isrc, album, year, genre, cover_url}
    source      TEXT,
    status      TEXT NOT NULL DEFAULT 'pending',  -- pending|confirmed|rejected
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS track_enrichment_proposals_user
    ON track_enrichment_proposals (user_id, status);

-- ── Public share links for tracks, albums and artists (canonical + TTL) ────
CREATE TABLE IF NOT EXISTS share_links (
    id           TEXT PRIMARY KEY,
    kind         TEXT NOT NULL CHECK (kind IN ('track', 'album', 'artist')),
    entity_id    BIGINT NOT NULL,
    created_by   INT REFERENCES users(id) ON DELETE SET NULL,
    title        TEXT NOT NULL,
    subtitle     TEXT,
    meta         TEXT,
    duration_sec INT,
    image_id     TEXT,
    links        JSONB NOT NULL DEFAULT '[]',
    clicks_total BIGINT NOT NULL DEFAULT 0,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    expires_at   TIMESTAMPTZ NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS share_links_entity_uidx ON share_links (kind, entity_id);
CREATE INDEX IF NOT EXISTS share_links_expires ON share_links (expires_at);

CREATE TABLE IF NOT EXISTS share_clicks (
    id         BIGSERIAL PRIMARY KEY,
    share_id   TEXT NOT NULL REFERENCES share_links(id) ON DELETE CASCADE,
    platform   TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS share_clicks_share_platform ON share_clicks (share_id, platform);

-- ── Admin-issued external website audio embed widgets (permanent until revoked) ──
CREATE TABLE IF NOT EXISTS embed_widgets (
    id              TEXT PRIMARY KEY,
    kind            TEXT NOT NULL CHECK (kind IN ('track', 'album', 'artist')),
    entity_id       BIGINT NOT NULL,
    title           TEXT NOT NULL,
    subtitle        TEXT,
    image_id        TEXT,
    theme           TEXT NOT NULL DEFAULT 'auto' CHECK (theme IN ('auto', 'dark', 'light')),
    layout          TEXT NOT NULL DEFAULT 'auto' CHECK (layout IN ('auto', 'compact', 'full')),
    allowed_origins TEXT[] NOT NULL DEFAULT '{}',
    edge_base_url   TEXT,
    active          BOOLEAN NOT NULL DEFAULT TRUE,
    created_by      INT REFERENCES users(id) ON DELETE SET NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS embed_widgets_entity ON embed_widgets (kind, entity_id);
CREATE INDEX IF NOT EXISTS embed_widgets_active ON embed_widgets (active, created_at DESC);

CREATE TABLE IF NOT EXISTS embed_widget_tracks (
    widget_id      TEXT NOT NULL REFERENCES embed_widgets(id) ON DELETE CASCADE,
    track_id       BIGINT NOT NULL REFERENCES tracks(id) ON DELETE CASCADE,
    position       INT NOT NULL,
    edge_warmed_at TIMESTAMPTZ,
    edge_warm_error TEXT,
    PRIMARY KEY (widget_id, track_id)
);
CREATE INDEX IF NOT EXISTS embed_widget_tracks_pos ON embed_widget_tracks (widget_id, position);
CREATE INDEX IF NOT EXISTS embed_widget_tracks_track ON embed_widget_tracks (track_id);
