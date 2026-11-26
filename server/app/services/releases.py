"""App releases for the in-app updater.

Two parity inputs (user choice):
  1. Admin upload: `POST /admin/api/app-releases` stores the APK under
     data/app-releases/{id}.apk and a row in app_releases.
  2. Manual drop: an operator scps `client.apk` + `client.json` into
     data/app-releases/inbox/ — no API call needed.

`latest()` serves whichever has the highest versionCode, so both ways are
equal citizens. APK bytes stay on disk; the DB only holds metadata.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

from .. import db
from ..config import settings

log = logging.getLogger("kwhale.releases")


def releases_dir() -> Path:
    d = Path(settings.data_dir) / "app-releases"
    d.mkdir(parents=True, exist_ok=True)
    return d


def inbox_dir() -> Path:
    d = releases_dir() / "inbox"
    d.mkdir(parents=True, exist_ok=True)
    return d


async def store_release(
    apk_bytes: bytes, version_code: int, version_name: str, notes: str | None
) -> dict[str, Any]:
    sha = hashlib.sha256(apk_bytes).hexdigest()
    rows = await db.execute_returning(
        """
        INSERT INTO app_releases (version_code, version_name, notes, file_path, sha256, size_bytes)
        VALUES (:vc, :vn, :n, '', :s, :sz)
        ON CONFLICT (version_code) DO UPDATE
        SET version_name = EXCLUDED.version_name,
            notes = EXCLUDED.notes,
            sha256 = EXCLUDED.sha256,
            size_bytes = EXCLUDED.size_bytes
        RETURNING id
        """,
        {"vc": version_code, "vn": version_name, "n": notes, "s": sha, "sz": len(apk_bytes)},
    )
    rid = rows[0]["id"]
    path = releases_dir() / f"{rid}.apk"
    path.write_bytes(apk_bytes)
    await db.execute(
        "UPDATE app_releases SET file_path = :p WHERE id = :i", {"p": str(path), "i": rid}
    )
    row = await db.fetch_one("SELECT * FROM app_releases WHERE id = :i", {"i": rid})
    return _to_dto(row)


def _to_dto(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": str(row["id"]),
        "versionCode": int(row["version_code"]),
        "versionName": row["version_name"],
        "notes": row.get("notes"),
        "sha256": row.get("sha256"),
        "sizeBytes": int(row.get("size_bytes") or 0),
        "createdAt": row["created_at"].isoformat() if row.get("created_at") else None,
    }


def _inbox_release() -> dict[str, Any] | None:
    """Manual parity input: inbox/client.json + inbox/client.apk (scp'ed)."""
    meta_path = inbox_dir() / "client.json"
    if not meta_path.is_file():
        return None
    try:
        meta = json.loads(meta_path.read_text())
    except Exception:  # noqa: BLE001
        log.warning("inbox client.json is not valid JSON")
        return None
    apk_path = inbox_dir() / meta.get("file", "client.apk")
    if not apk_path.is_file():
        log.warning("inbox client.json points to a missing APK: %s", apk_path)
        return None
    data = apk_path.read_bytes()
    return {
        "id": f"inbox:{apk_path.name}",
        "versionCode": int(meta.get("versionCode") or 0),
        "versionName": str(meta.get("versionName") or ""),
        "notes": meta.get("notes"),
        "sha256": hashlib.sha256(data).hexdigest(),
        "sizeBytes": len(data),
        "createdAt": None,
        "_path": str(apk_path),
    }


async def latest() -> dict[str, Any] | None:
    """Newest release across both inputs; null when nothing is published."""
    row = await db.fetch_one(
        "SELECT * FROM app_releases ORDER BY version_code DESC LIMIT 1"
    )
    best = _to_dto(row) if row else None
    if best is not None:
        best["_path"] = row["file_path"]
    inbox = _inbox_release()
    if inbox and (best is None or inbox["versionCode"] > best["versionCode"]):
        best = inbox
    return best


async def apk_path_for(release: dict[str, Any]) -> Path | None:
    path = Path(release.get("_path") or "")
    return path if path.is_file() else None


async def list_releases() -> list[dict[str, Any]]:
    rows = await db.fetch_all("SELECT * FROM app_releases ORDER BY version_code DESC LIMIT 50")
    return [_to_dto(r) for r in rows]
