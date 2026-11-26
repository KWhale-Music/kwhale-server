"""Application settings, env-driven."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path


def _load_dotenv() -> None:
    env_path = Path(__file__).resolve().parents[2] / ".env"
    if not env_path.is_file():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())


_load_dotenv()


def _default_music_dir() -> str:
    if Path("/data/music").exists():
        return "/data/music"
    local_dir = Path(__file__).resolve().parents[2] / "data" / "music"
    local_dir.mkdir(parents=True, exist_ok=True)
    return str(local_dir)


def _env(key: str, default: str = "") -> str:
    return os.environ.get(key, default).strip()


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.environ.get(key, "").strip() or default)
    except ValueError:
        return default


def _env_bool(key: str, default: bool = False) -> bool:
    raw = os.environ.get(key)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _source_proxies() -> dict[str, str]:
    """SOURCE_PROXY_<CONNECTOR> env entries -> {connector: proxy-url}."""
    out: dict[str, str] = {}
    prefix = "SOURCE_PROXY_"
    for key, value in os.environ.items():
        if key.startswith(prefix) and key != f"{prefix}DEFAULT" and value.strip():
            out[key[len(prefix) :].lower()] = value.strip()
    return out


@dataclass(frozen=True)
class Settings:
    # ── core ──────────────────────────────────────────────────────────────
    database_url: str = field(
        default_factory=lambda: _env(
            "DATABASE_URL", "postgresql+asyncpg://kwhale:kwhale_v3_pg@127.0.0.1:5433/kwhale"
        )
    )
    # API keeps a wide pool; the indexer workers must keep theirs tiny — five
    # workers x 20 connections exhausted Postgres max_connections=100 and made
    # trivial API queries wait seconds for a free backend.
    db_pool_size: int = field(default_factory=lambda: _env_int("DB_POOL_SIZE", 20))
    db_max_overflow: int = field(default_factory=lambda: _env_int("DB_MAX_OVERFLOW", 10))
    # Per-statement watchdog for the API pool (one analytic query must not pin
    # a backend). 0 = unlimited — repair/maintenance CLIs set this because their
    # full-table scans legitimately exceed it.
    db_statement_timeout_ms: int = field(
        default_factory=lambda: _env_int("DB_STATEMENT_TIMEOUT_MS", 15000)
    )
    # The downloader is I/O-heavy and runs its own asyncio loop. In the API
    # process it competes with request handling (one loop), so production runs
    # it in a dedicated container and turns the in-process runner off.
    downloader_enabled: bool = field(
        default_factory=lambda: _env_bool("DOWNLOADER_ENABLED", True)
    )
    jwt_secret: str = field(default_factory=lambda: _env("JWT_SECRET", "dev-secret-change-me"))
    jwt_expire_hours: int = field(default_factory=lambda: _env_int("JWT_EXPIRE_HOURS", 24 * 30))
    jwt_access_minutes: int = field(default_factory=lambda: _env_int("JWT_ACCESS_MINUTES", 15))
    refresh_token_days: int = field(default_factory=lambda: _env_int("REFRESH_TOKEN_DAYS", 30))

    # seed user created at startup when missing
    seed_username: str = field(default_factory=lambda: _env("SEED_USERNAME", "admin"))
    seed_password: str = field(default_factory=lambda: _env("SEED_PASSWORD", ""))
    seed_email: str = field(default_factory=lambda: _env("SEED_EMAIL", "admin@local"))
    seed_display_name: str = field(default_factory=lambda: _env("SEED_DISPLAY_NAME", "admin"))

    # ── security ────────────────────────────────────────────────────────────
    # CORS: this is a Bearer-token API (no cookies), so allow_credentials stays
    # off. Origins are configurable (comma-separated), default "*" for LAN dev.
    cors_origins: list[str] = field(
        default_factory=lambda: [
            o.strip() for o in _env("CORS_ORIGINS", "*").split(",") if o.strip()
        ]
    )
    # Fail fast on weak/dev secrets at boot when true (APP_ENV=production or
    # STRICT_SECRETS=1). Otherwise we only warn.
    strict_secrets: bool = field(
        default_factory=lambda: _env_bool("STRICT_SECRETS", False)
        or _env("APP_ENV", "").strip().lower() == "production"
    )

    # ── storage ───────────────────────────────────────────────────────────
    music_dir: Path = field(default_factory=lambda: Path(_env("MUSIC_DIR", _default_music_dir())))
    # app-release APKs for the in-app updater (inside the data mount)
    data_dir: Path = field(default_factory=lambda: Path(_env("DATA_DIR", _default_music_dir())).parent)
    # ephemeral tracks not seen in search results for N days get GC-ed
    ephemeral_ttl_days: int = field(default_factory=lambda: _env_int("EPHEMERAL_TTL_DAYS", 14))
    # last user unlikes a downloaded track -> file removed after N days
    unlike_grace_days: int = field(default_factory=lambda: _env_int("UNLIKE_GRACE_DAYS", 14))

    # ── indexer: audio embeddings ─────────────────────────────────────────
    # Bounded batch pipeline for `kind='audio'` jobs: a night run over the whole
    # library must stay polite to Postgres and never touch the API loop.
    indexer_audio_workers: int = field(
        default_factory=lambda: _env_int("INDEXER_AUDIO_WORKERS", 2)
    )
    indexer_audio_batch: int = field(default_factory=lambda: _env_int("INDEXER_AUDIO_BATCH", 100))

    # ── connectors ────────────────────────────────────────────────────────
    # ICM (byicloud.online partner API = Apple Music mirror)
    icm_partner_key: str = field(default_factory=lambda: _env("ICM_PARTNER_KEY"))
    icm_base_url: str = field(
        default_factory=lambda: _env("ICM_BASE_URL", "https://byicloud.online").rstrip("/")
    )
    icm_partner_user_id: str = field(default_factory=lambda: _env("ICM_PARTNER_USER_ID"))
    icm_region: str = field(default_factory=lambda: _env("ICM_REGION", "us"))
    icm_fallback_region: str = field(default_factory=lambda: _env("ICM_FALLBACK_REGION", "ru"))
    icm_quality: str = field(default_factory=lambda: _env("ICM_QUALITY", "256K"))
    # keepalive period in hours (ICM invalidates stale keys; >=1 request per day)
    icm_keepalive_hours: int = field(default_factory=lambda: _env_int("ICM_KEEPALIVE_HOURS", 24))

    # Yandex Music
    yandex_token: str = field(default_factory=lambda: _env("YANDEX_MUSIC_TOKEN"))

    # DeepSeek (LLM scoring patches for the recommendation layer)
    deepseek_api_key: str = field(default_factory=lambda: _env("DEEPSEEK_API_KEY"))
    deepseek_base_url: str = field(
        default_factory=lambda: _env("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/")
    )
    deepseek_model: str = field(default_factory=lambda: _env("DEEPSEEK_MODEL", "deepseek-chat"))
    deepseek_timeout: float = field(default_factory=lambda: float(_env("DEEPSEEK_TIMEOUT", "25")))

    # Giga text embeddings (assistant-embeddings service, 1024-dim)
    giga_embeddings_url: str = field(
        default_factory=lambda: _env("GIGA_EMBEDDINGS_URL", "http://embeddings:8096/v1/embeddings")
    )
    giga_embeddings_model: str = field(
        default_factory=lambda: _env("GIGA_EMBEDDINGS_MODEL", "giga-embeddings-instruct-480m-0826")
    )
    giga_embeddings_timeout: float = field(
        default_factory=lambda: float(_env("GIGA_EMBEDDINGS_TIMEOUT", "30"))
    )

    # SoundCloud
    soundcloud_client_id: str = field(default_factory=lambda: _env("SOUNDCLOUD_CLIENT_ID"))

    # ── proxy / network routing (modular across machines) ─────────────────
    # Global switch: set PROXY_ENABLED=false when running on a host with system VPN
    proxy_enabled: bool = field(default_factory=lambda: _env_bool("PROXY_ENABLED", True))
    # Default proxy for foreign sources / yt-dlp / Shazam (empty = direct)
    source_proxy_default: str = field(default_factory=lambda: _env("SOURCE_PROXY_DEFAULT", ""))
    vpn_proxy: str = field(default_factory=lambda: _env("VPN_PROXY", ""))

    # ── source accounts (web-ui /admin) ──────────────────────────────────
    # AES-GCM key for source account secrets at rest; derived from JWT_SECRET
    # (dev fallback) when unset
    accounts_enc_key: str = field(default_factory=lambda: _env("ACCOUNTS_ENC_KEY"))
    # per-connector proxy routing: SOURCE_PROXY_<CONNECTOR>=socks5h://… or 'direct'
    source_proxies: dict[str, str] = field(default_factory=_source_proxies)

    # ── manual pipeline ────────────────────────────────────────────────────
    # AcoustID client key (public, from kwhale-v2)
    acoustid_client: str = field(default_factory=lambda: _env("ACOUSTID_CLIENT", "v8pQ6oyB"))
    # Shazam proxy (empty = uses default proxy if enabled, or direct)
    shazam_proxy: str = field(default_factory=lambda: _env("SHAZAM_PROXY", ""))
    # dropbox file considered stable when mtime older than N seconds
    dropbox_stable_seconds: int = field(
        default_factory=lambda: _env_int("DROPBOX_STABLE_SECONDS", 30)
    )

    # ── search/stream/cache tuning ─────────────────────────────────────────
    search_limit: int = field(default_factory=lambda: _env_int("SEARCH_LIMIT", 30))
    connector_timeout: float = field(default_factory=lambda: float(_env("CONNECTOR_TIMEOUT", "12")))
    search_cache_ttl_hours: int = field(
        default_factory=lambda: _env_int("SEARCH_CACHE_TTL_HOURS", 6)
    )
    # Covers referenced only by search results are disposable; liked/local
    # covers are pinned and never expire.
    image_cache_ttl_days: int = field(
        default_factory=lambda: _env_int("IMAGE_CACHE_TTL_DAYS", 14)
    )
    artist_sync_ttl_days: int = field(
        default_factory=lambda: _env_int("ARTIST_SYNC_TTL_DAYS", 2)
    )
    stream_chunk_size: int = field(default_factory=lambda: _env_int("STREAM_CHUNK_SIZE", 64 * 1024))
    # LRU disk cache for played tracks (instant Range seek without permanent save)
    stream_cache_max_mb: int = field(default_factory=lambda: _env_int("STREAM_CACHE_MAX_MB", 200))
    stream_cache_ttl_hours: float = field(
        default_factory=lambda: float(_env("STREAM_CACHE_TTL_HOURS", "2.0") or 2.0)
    )
    # Raw events retention (§10)
    events_retention_days: int = field(
        default_factory=lambda: _env_int("EVENTS_RETENTION_DAYS", 180)
    )
    # Public share links sliding TTL (days)
    share_link_ttl_days: int = field(
        default_factory=lambda: _env_int("SHARE_LINK_TTL_DAYS", 7)
    )
    # Optional public Edge origin for warming cached media slices on embed issue
    embed_edge_base_url: str = field(
        default_factory=lambda: (
            _env("EMBED_EDGE_BASE_URL") or _env("EDGE_WARM_BASE_URL")
        ).rstrip("/")
    )
    embed_warm_concurrency: int = field(
        default_factory=lambda: max(1, _env_int("EMBED_WARM_CONCURRENCY", 2))
    )


settings = Settings()


def validate_security(s: Settings) -> None:
    """Fail fast on insecure secrets in strict/production mode; warn otherwise."""
    log = logging.getLogger("kwhale.config")
    weak_jwt = (not s.jwt_secret) or s.jwt_secret == "dev-secret-change-me"
    weak_enc = not s.accounts_enc_key
    if s.strict_secrets:
        if weak_jwt:
            raise RuntimeError(
                "JWT_SECRET отсутствует или это dev-дефолт — задайте реальный секрет "
                "(APP_ENV=production / STRICT_SECRETS=1 требуют его)."
            )
        if weak_enc:
            raise RuntimeError(
                "ACCOUNTS_ENC_KEY не задан — задайте реальный ключ "
                "(APP_ENV=production / STRICT_SECRETS=1 требуют его)."
            )
    else:
        if weak_jwt:
            log.warning("JWT_SECRET — dev-дефолт; перед production задайте реальный секрет")
        if weak_enc:
            log.warning("ACCOUNTS_ENC_KEY не задан; секреты источников шифруются ключом от JWT")
