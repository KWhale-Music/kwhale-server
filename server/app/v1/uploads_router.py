"""KWhale v1 own-track uploads."""

from __future__ import annotations

import logging
import uuid
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, UploadFile

from ..auth import current_user
from ..errors import ProblemError
from ..services import enrichment, manual
from ..services.library import staging_dir
from .schemas import Track
from .serializers import track_dto

log = logging.getLogger("kwhale.uploads")

router = APIRouter(prefix="/uploads", tags=["uploads"])

MAX_UPLOAD = 100 * 1024 * 1024


@router.post("/tracks", response_model=Track, status_code=201)
async def upload_track(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    title: str | None = Form(None),
    artist: str | None = Form(None),
    album: str | None = Form(None),
    user: dict = Depends(current_user),
) -> Track:
    if not file.filename:
        raise ProblemError(400, "validation_failed", "filename is required")
    suffix = Path(file.filename).suffix.lower()
    if suffix not in manual._AUDIO_EXTS:
        raise ProblemError(400, "validation_failed", f"unsupported audio type: {suffix or 'none'}")
    upload_id = uuid.uuid4().hex[:12]
    dest_dir = staging_dir() / "manual" / upload_id
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"upload{suffix}"
    total = 0
    try:
        with dest.open("wb") as f:
            while chunk := await file.read(1024 * 1024):
                total += len(chunk)
                if total > MAX_UPLOAD:
                    raise ProblemError(413, "upload_too_large", "file is larger than 100 MB")
                f.write(chunk)
    except ProblemError:
        dest.unlink(missing_ok=True)
        dest_dir.rmdir()
        raise

    hint = {k: v for k, v in {"title": title, "artist": artist, "album": album}.items() if v}
    try:
        track_id = await manual.process_manual_file(
            dest, hint=hint or None, user_id=user["id"]
        )
    except Exception as e:  # noqa: BLE001
        log.exception("upload processing failed for user %s", user["id"])
        raise ProblemError(500, "internal_error", "processing failed") from e
    if track_id is None:
        raise ProblemError(
            422,
            "validation_failed",
            "could not resolve metadata (no tags/fingerprint match); quarantined",
        )
    dto = await track_dto(track_id, user["id"])
    if dto is None:
        raise ProblemError(500, "internal_error", "uploaded track vanished")
    # Enrich metadata from the connected sources in the background; the user
    # confirms any proposal from the profile before it is applied.
    background_tasks.add_task(enrichment.enrich_track, track_id, user["id"])
    return dto


@router.get("/enrichments")
async def list_enrichments(user: dict = Depends(current_user)) -> dict:
    return {"items": await enrichment.list_pending(user["id"])}


@router.post("/enrichments/{proposal_id}/confirm")
async def confirm_enrichment(proposal_id: int, user: dict = Depends(current_user)) -> dict:
    if not await enrichment.confirm(proposal_id, user["id"]):
        raise ProblemError(404, "not_found", "enrichment proposal not found or already handled")
    return {"ok": True, "id": proposal_id, "status": "confirmed"}


@router.post("/enrichments/{proposal_id}/reject")
async def reject_enrichment(proposal_id: int, user: dict = Depends(current_user)) -> dict:
    if not await enrichment.reject(proposal_id, user["id"]):
        raise ProblemError(404, "not_found", "enrichment proposal not found or already handled")
    return {"ok": True, "id": proposal_id, "status": "rejected"}
