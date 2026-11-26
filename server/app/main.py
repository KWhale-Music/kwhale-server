"""KWhale-v3 API entrypoint — public KWhale Core v1 at /v1."""

import asyncio
import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from . import db
from .auth import ensure_seed_user
from .config import settings, validate_security
from .errors import (
    ProblemError,
    http_exception_handler,
    problem_body,
    problem_handler,
    validation_exception_handler,
)
from .public_embed import router as public_embed_router
from .public_share import router as public_share_router
from .services import downloader, maintenance
from .services.entity_upsert import ensure_favorite_entities
from .services.library import migrate_flat_files
from .services.rec.scoring import seed_profiles
from .v1 import api_v1
from .v1.admin_router import router as admin_api_router
from .v1.admin_ui import router as admin_ui_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("kwhale")


async def _warm_entity_indexes() -> None:
    """Build the artist/album lookup maps before the first user search.

    The build normalizes ~56k names; running it inside the first request blocked
    the event loop for ~1.5 s. dto._artist_id_index/_album_id_index offload the
    CPU part to a thread, so this task is cheap and keeps cold searches fast.
    """
    from .services.dto import _album_id_index, _artist_id_index

    try:
        await _artist_id_index()
        await _album_id_index()
        log.info("entity id indexes warmed")
    except Exception:
        log.exception("entity id index warmup failed")


@asynccontextmanager
async def lifespan(app: FastAPI):
    from . import httpcore

    validate_security(settings)  # fail fast on weak secrets before doing anything
    await db.apply_schema()
    await httpcore.load_proxy_settings()
    await ensure_seed_user()
    try:
        await seed_profiles()
    except Exception:
        log.exception("scoring profile seed failed")
    if settings.downloader_enabled:
        await downloader.start_runner()
    await maintenance.start()
    await migrate_flat_files()
    # materialise artist/album rows for liked tracks (navigation cards)
    try:
        await ensure_favorite_entities()
    except Exception:
        log.exception("favorite entity backfill failed")
    asyncio.create_task(_warm_entity_indexes())
    log.info("kwhale-v3 up (music_dir=%s)", settings.music_dir)
    yield
    from .connectors import get_connectors

    for conn in get_connectors().values():
        if hasattr(conn, "aclose"):
            await conn.aclose()
    from . import httpcore
    from .services import embed_service, stream_service

    await embed_service.aclose_background_tasks()
    await stream_service.aclose_clients()
    await httpcore.aclose_all()
    await db.close_engine()


app = FastAPI(title="KWhale-v3 API", version="1.0.0", lifespan=lifespan)

app.add_middleware(GZipMiddleware, minimum_size=512)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=False,  # Bearer-token API — no cookies/credentials to leak
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_v1, prefix="/v1")
app.include_router(admin_api_router)
app.include_router(admin_ui_router)
app.include_router(public_embed_router)
app.include_router(public_share_router)

app.add_exception_handler(ProblemError, problem_handler)  # type: ignore[arg-type]
app.add_exception_handler(StarletteHTTPException, http_exception_handler)  # type: ignore[arg-type]
app.add_exception_handler(RequestValidationError, validation_exception_handler)  # type: ignore[arg-type]


@app.middleware("http")
async def add_response_time(request: Request, call_next):
    started = time.perf_counter()
    response = await call_next(request)
    response.headers["X-Response-Time"] = f"{(time.perf_counter() - started) * 1000:.0f}ms"
    if request.url.path.startswith("/v1/"):
        if "cache-control" not in response.headers:
            response.headers["Cache-Control"] = "private, no-store"
        cc = response.headers.get("cache-control", "").lower()
        if "public" not in cc and request.headers.get("authorization"):
            vary_parts = [v.strip() for v in response.headers.get("vary", "").split(",") if v.strip()]
            if not any(v.lower() == "authorization" for v in vary_parts):
                vary_parts.append("Authorization")
            response.headers["Vary"] = ", ".join(vary_parts)
    return response


@app.exception_handler(Exception)
async def unhandled_error_handler(request: Request, exc: Exception):
    log.exception("unhandled error on %s %s", request.method, request.url.path)
    if request.url.path.startswith("/v1"):
        return JSONResponse(
            status_code=500,
            content=problem_body(500, "internal_error", "Internal server error", instance=request.url.path),
            media_type="application/problem+json",
        )
    return JSONResponse(
        status_code=500,
        content={"error": {"code": "internal", "message": "internal server error"}},
    )


@app.get("/healthz", tags=["health"])
async def healthz():
    return {"status": "ok"}
