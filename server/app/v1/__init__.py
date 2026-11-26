"""Canonical KWhale Core v1 public API."""

from __future__ import annotations

from fastapi import APIRouter

from . import (
    auth_router,
    catalog_router,
    images_router,
    misc_router,
    playlists_router,
    search_router,
    share_router,
    uploads_router,
)

api_v1 = APIRouter()
api_v1.include_router(auth_router.router)
api_v1.include_router(auth_router.me_router)
api_v1.include_router(search_router.router)
api_v1.include_router(catalog_router.router)
api_v1.include_router(playlists_router.router)
api_v1.include_router(misc_router.router)
api_v1.include_router(uploads_router.router)
api_v1.include_router(images_router.router)
api_v1.include_router(share_router.router)

__all__ = ["api_v1"]
