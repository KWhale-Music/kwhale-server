"""Source accounts: server-wide per-connector credentials chosen by an admin.

One row per connector in `source_accounts`. Secrets (tokens, cookies) are
stored AES-GCM encrypted (services/secrets.py); `profile` holds only
non-secret display info (name, country, subscription). A connector takes part
in search fan-out only when it is enabled and usable — the user picks services
by connecting them (web-ui /admin).

Two credential kinds:
  * account — payload pasted/entered via UI (deezer/zvuk/vk/qobuz/tidal)
  * env     — configured through env keys at deploy (icm/yandex/soundcloud/ytmusic);
              the row only carries the enabled toggle and status
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from .. import db
from .secrets import decrypt_text, encrypt_text

log = logging.getLogger("kwhale.accounts")

# status values
S_CONNECTED = "connected"
S_EXPIRED = "expired"
S_ERROR = "error"
S_DISABLED = "disabled"

# UI/registry metadata per source. `fields` = secrets the connect form asks for.
SOURCE_SPECS: dict[str, dict[str, Any]] = {
    "icm": {
        "label": "ICM (Apple Music)",
        "auth": "token",
        "optional_fields": ["partner_key"],
        "field_help": {"partner_key": "ICM_PARTNER_KEY (оставьте пустым, если задан в .env)"},
        "env_keys": ["ICM_PARTNER_KEY"],
        "help": "partner-ключ ICM из админки или .env (ICM_PARTNER_KEY)",
    },
    "yandex": {
        "label": "Яндекс Музыка",
        "auth": "token",
        "optional_fields": ["token"],
        "field_help": {"token": "OAuth-токен (оставьте пустым, если задан в .env)"},
        "env_keys": ["YANDEX_MUSIC_TOKEN"],
        "playback": "FLAC / MP3 320 с подпиской Плюс",
        "help": "OAuth-токен из админки или .env (YANDEX_MUSIC_TOKEN)",
    },
    "soundcloud": {
        "label": "SoundCloud",
        "auth": "env",
        "env_keys": ["SOUNDCLOUD_CLIENT_ID"],
        "optional": True,
        "help": "работает автоматически (авто-скрапинг client_id + yt-dlp fallback)",
    },
    "ytmusic": {
        "label": "YouTube Music",
        "auth": "env",
        "env_keys": [],
        "optional": True,
        "help": "поиск InnerTube + поток через yt-dlp, ключи не нужны",
    },
    "deezer": {
        "label": "Deezer",
        "auth": "cookie",
        "fields": ["arl"],
        "field_help": {"arl": "cookie arl с deezer.com (DevTools → Application → Cookies)"},
        "playback": "MP3 320 — бесплатно; FLAC — подписка Hi-Fi",
        "help": "вставьте cookie arl — остальное сервер получит сам",
    },
    "zvuk": {
        "label": "Звук (SberZvuk)",
        "auth": "password",
        "fields": ["email", "password"],
        "alt_fields": ["token"],
        "field_help": {
            "email": "email аккаунта zvuk.com",
            "password": "пароль",
            "token": "или готовый x-auth-token (32 символа)",
        },
        "playback": "каталог по подписке; поиск доступен всем",
        "help": "логин email/пароль или вставьте токен",
    },
    "vk": {
        "label": "ВКонтакте",
        "auth": "cookie",
        "fields": ["p", "remixnttpid"],
        "field_help": {
            "p": "cookie p с login.vk.ru",
            "remixnttpid": "cookie remixnttpid с vk.ru",
        },
        "playback": "поиск и стрим по доступным аудиозаписям",
        "help": "войдите в VK в браузере и скопируйте две cookie",
    },
    "qobuz": {
        "label": "Qobuz",
        "auth": "token",
        "fields": ["token"],
        "optional_fields": ["app_id", "app_secret"],
        "field_help": {
            "token": "x-user-auth-token (после входа на play.qobuz.com)",
            "app_id": "app_id — опционально, иначе добудем из bundle.js",
            "app_secret": "app_secret — опционально",
        },
        "playback": "стриминг — только с подпиской (иначе каталог и поиск)",
        "help": "без подписки доступны поиск и метаданные (ISRC)",
    },
    "tidal": {
        "label": "Tidal",
        "auth": "token",
        "fields": ["refresh_token"],
        "optional_fields": ["license_country"],
        "field_help": {
            "refresh_token": "OAuth2 refresh token",
            "license_country": "ISO-код страны лицензии (дефолт US)",
        },
        "playback": "стриминг — только Premium (иначе каталог и поиск)",
        "help": "без Premium доступны поиск и метаданные (ISRC)",
    },
}

# sources usable without any user credential (beyond deploy-time env)
NO_LOGIN = {"soundcloud", "ytmusic"}


@dataclass
class SourceAccount:
    connector: str
    kind: str = "account"
    payload: dict[str, Any] = field(default_factory=dict)
    profile: dict[str, Any] = field(default_factory=dict)
    status: str = S_CONNECTED
    enabled: bool = True

    def redacted(self) -> dict[str, Any]:
        return {
            "connector": self.connector,
            "kind": self.kind,
            "status": self.status,
            "enabled": self.enabled,
            "profile": self.profile,
            "has_payload": bool(self.payload),
        }


def _row_to_account(row: dict[str, Any]) -> SourceAccount:
    payload: dict[str, Any] = {}
    if row.get("payload_enc"):
        try:
            payload = json.loads(decrypt_text(row["payload_enc"]))
        except Exception as e:  # noqa: BLE001 — corrupt row must not kill listing
            log.error("account %s: payload decrypt failed: %s", row["connector"], e)
    profile = row.get("profile") or {}
    if isinstance(profile, str):
        profile = json.loads(profile)
    return SourceAccount(
        connector=row["connector"],
        kind=row.get("kind") or "account",
        payload=payload,
        profile=profile,
        status=row.get("status") or S_CONNECTED,
        enabled=bool(row.get("enabled", True)),
    )


async def get_account(connector: str) -> SourceAccount | None:
    row = await db.fetch_one(
        "SELECT * FROM source_accounts WHERE connector = :c", {"c": connector}
    )
    return _row_to_account(row) if row else None


async def get_payload(connector: str) -> dict[str, Any] | None:
    acc = await get_account(connector)
    return acc.payload if acc and acc.payload else None


async def save_account(
    connector: str,
    payload: dict[str, Any],
    profile: dict[str, Any] | None = None,
    status: str = S_CONNECTED,
    kind: str = "account",
) -> None:
    enc = encrypt_text(json.dumps(payload)) if payload else None
    await db.execute(
        """
        INSERT INTO source_accounts (connector, kind, payload_enc, profile, status, enabled, updated_at)
        VALUES (:c, :k, :p, :pr, :s, TRUE, NOW())
        ON CONFLICT (connector) DO UPDATE
        SET kind = EXCLUDED.kind,
            payload_enc = COALESCE(EXCLUDED.payload_enc, source_accounts.payload_enc),
            profile = EXCLUDED.profile,
            status = EXCLUDED.status,
            enabled = TRUE,
            updated_at = NOW()
        """,
        {
            "c": connector,
            "k": kind,
            "p": enc,
            "pr": json.dumps(profile or {}),
            "s": status,
        },
    )


async def update_payload(connector: str, updates: dict[str, Any]) -> None:
    """Merge token refreshes into the stored payload (re-encrypted)."""
    acc = await get_account(connector)
    if acc is None:
        return
    merged = {**acc.payload, **{k: v for k, v in updates.items() if v is not None}}
    await save_account(
        connector,
        merged,
        profile=acc.profile,
        status=acc.status,
        kind=acc.kind,
    )


async def set_profile(
    connector: str,
    profile: dict[str, Any] | None = None,
    status: str | None = None,
) -> None:
    acc = await get_account(connector)
    if acc is None:
        return
    await save_account(
        connector,
        acc.payload,
        profile=profile if profile is not None else acc.profile,
        status=status or acc.status,
        kind=acc.kind,
    )


async def set_enabled(connector: str, enabled: bool) -> None:
    row = await db.fetch_one(
        "SELECT connector FROM source_accounts WHERE connector = :c", {"c": connector}
    )
    if row is None:
        # env-configured source without a row: create a toggle-only row
        kind = "env" if (SOURCE_SPECS.get(connector) or {}).get("auth") == "env" else "account"
        await db.execute(
            """
            INSERT INTO source_accounts (connector, kind, payload_enc, profile, status, enabled, updated_at)
            VALUES (:c, :k, NULL, '{}'::jsonb, :s, :e, NOW())
            ON CONFLICT (connector) DO UPDATE SET enabled = EXCLUDED.enabled, updated_at = NOW()
            """,
            {
                "c": connector,
                "k": kind,
                "s": S_CONNECTED if enabled else S_DISABLED,
                "e": enabled,
            },
        )
        return
    await db.execute(
        "UPDATE source_accounts SET enabled = :e, updated_at = NOW() WHERE connector = :c",
        {"e": enabled, "c": connector},
    )


async def disconnect(connector: str) -> None:
    await db.execute("DELETE FROM source_accounts WHERE connector = :c", {"c": connector})


async def list_raw() -> list[SourceAccount]:
    rows = await db.fetch_all("SELECT * FROM source_accounts ORDER BY connector")
    return [_row_to_account(r) for r in rows]


async def enabled_connectors() -> set[str]:
    """Connector names allowed to take part in search fan-out."""
    rows = await db.fetch_all(
        "SELECT connector, enabled FROM source_accounts WHERE enabled = FALSE"
    )
    disabled = {r["connector"] for r in rows}
    return {name for name in SOURCE_SPECS if name not in disabled}


async def admin_view() -> list[dict[str, Any]]:
    """Full source list for /admin: spec + account state (no secrets)."""
    import os

    from .. import httpcore
    from ..connectors import get_connectors

    accounts = {a.connector: a for a in await list_raw()}
    connectors = get_connectors()
    out: list[dict[str, Any]] = []
    for name, spec in SOURCE_SPECS.items():
        acc = accounts.get(name)
        conn = connectors.get(name)
        env_keys = spec.get("env_keys") or []
        env_ready = bool(env_keys) and all(os.environ.get(k, "").strip() for k in env_keys)
        if spec.get("optional") and not env_keys:
            env_ready = True
        has_db_payload = acc is not None and bool(acc.payload)
        is_avail = await conn.is_available() if conn else False
        if acc is not None and not acc.enabled:
            status = S_DISABLED
        elif has_db_payload:
            status = acc.status if acc else S_CONNECTED
        elif env_ready or is_avail:
            status = S_CONNECTED
        elif spec.get("auth") == "env":
            status = "missing_key"
        else:
            status = "disconnected"
        out.append(
            {
                "name": name,
                "label": spec.get("label", name),
                "auth": spec.get("auth"),
                "fields": spec.get("fields") or [],
                "optional_fields": spec.get("optional_fields") or [],
                "alt_fields": spec.get("alt_fields") or [],
                "field_help": spec.get("field_help") or {},
                "help": spec.get("help", ""),
                "playback": spec.get("playback", ""),
                "connected": has_db_payload or env_ready,
                "enabled": bool(acc.enabled) if acc else True,
                "status": status,
                "profile": acc.profile if acc else {},
                "env_ready": env_ready,
                "available": is_avail,
                "effectiveProxy": httpcore.proxy_for(name) or "direct",
            }
        )
    return out
