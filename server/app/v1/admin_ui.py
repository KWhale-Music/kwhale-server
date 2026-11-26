"""Admin web UI shell. Authentication happens in JS via /v1/auth/login."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

router = APIRouter(include_in_schema=False)

_HTML_PATH = Path(__file__).resolve().parent.parent / "static" / "admin.html"


@router.get("/admin", response_class=HTMLResponse)
async def admin_page() -> HTMLResponse:
    return HTMLResponse(_HTML_PATH.read_text(encoding="utf-8"))
