"""Image cache: external cover URLs are hidden behind stable local ids."""

from __future__ import annotations

import asyncio
import hashlib
import io
from typing import Any

from fastapi import Response
from PIL import Image, ImageDraw

from .. import db
from ..errors import ProblemError
from .schemas import ImageRef

PLACEHOLDER_ID = "placeholder"
_SIZES = {"xs": 144, "sm": 300, "md": 600, "lg": 1000}
_MIME_BY_FORMAT = {"JPEG": "image/jpeg", "PNG": "image/png", "WEBP": "image/webp"}


def image_id_for_url(url: str | None) -> str:
    if not url:
        return PLACEHOLDER_ID
    return hashlib.sha256(url.encode()).hexdigest()[:32]


async def image_ref(url: str | None) -> ImageRef:
    image_id = image_id_for_url(url)
    if image_id != PLACEHOLDER_ID and url:
        await db.execute(
            """
            INSERT INTO images (id, source_url) VALUES (:i, :u)
            ON CONFLICT (id) DO NOTHING
            """,
            {"i": image_id, "u": url},
        )
    return ImageRef(id=image_id)


async def image_refs_batch(urls: list[str | None]) -> dict[str, ImageRef]:
    """Batch-register cover URLs in a single DB round-trip and return url -> ImageRef."""
    pairs: dict[str, str] = {}
    for u in urls:
        if u:
            iid = image_id_for_url(u)
            if iid != PLACEHOLDER_ID:
                pairs[iid] = u
    if pairs:
        iids = list(pairs.keys())
        surls = [pairs[k] for k in iids]
        await db.execute(
            """
            INSERT INTO images (id, source_url)
            SELECT unnest(CAST(:ids AS text[])), unnest(CAST(:urls AS text[]))
            ON CONFLICT (id) DO NOTHING
            """,
            {"ids": iids, "urls": surls},
        )
    out: dict[str, ImageRef] = {}
    for u in urls:
        if u:
            out[u] = ImageRef(id=image_id_for_url(u))
    return out


async def store_uploaded_image(data: bytes, mime: str) -> str:
    """Persist a user-uploaded playlist cover and return its image id."""
    image_id = "u" + hashlib.sha256(data).hexdigest()[:31]
    width = height = None
    try:
        with Image.open(io.BytesIO(data)) as im:
            width, height = im.size
    except Exception:  # noqa: BLE001
        pass
    await db.execute(
        """
        INSERT INTO images
            (id, source_url, mime, data, width, height, fetched_at, created_at, last_accessed_at)
        VALUES
            (:i, :u, :m, :d, :w, :h, NOW(), NOW(), NOW())
        ON CONFLICT (id) DO UPDATE
        SET mime = EXCLUDED.mime, data = EXCLUDED.data,
            width = EXCLUDED.width, height = EXCLUDED.height,
            fetched_at = NOW(), last_accessed_at = NOW()
        """,
        {
            "i": image_id,
            "u": f"upload://{image_id}",
            "m": mime,
            "d": data,
            "w": width,
            "h": height,
        },
    )
    return image_id


def placeholder_ref() -> ImageRef:
    return ImageRef(id=PLACEHOLDER_ID)


def _placeholder(size: str) -> tuple[bytes, str, str]:
    edge = _SIZES.get(size, _SIZES["md"])
    if size == "original":
        edge = _SIZES["lg"]
    img = Image.new("RGB", (edge, edge), (24, 28, 34))
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle(
        (edge * 0.18, edge * 0.18, edge * 0.82, edge * 0.82),
        radius=edge * 0.08,
        fill=(35, 42, 51),
    )
    draw.text((edge * 0.43, edge * 0.34), "K", fill=(120, 180, 255))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    data = buf.getvalue()
    return data, "image/png", hashlib.sha256(data).hexdigest()


def _resize(data: bytes, size: str) -> tuple[bytes, str, str]:
    if size == "original":
        return data, "", hashlib.sha256(data).hexdigest()
    edge = _SIZES.get(size, _SIZES["lg"])
    with Image.open(io.BytesIO(data)) as src:
        im = src.convert("RGB")
        w, h = im.size
        if w <= edge and h <= edge:
            out = io.BytesIO()
            im.save(out, format="WEBP", quality=92, method=4)
        else:
            im.thumbnail((edge, edge), Image.Resampling.LANCZOS)
            out = io.BytesIO()
            im.save(out, format="WEBP", quality=92, method=4)
    resized = out.getvalue()
    return resized, "image/webp", hashlib.sha256(resized).hexdigest()


async def _load(image_id: str) -> dict[str, Any]:
    row = await db.fetch_one("SELECT * FROM images WHERE id = :i", {"i": image_id})
    if row is None:
        raise ProblemError(404, "not_found", "Image not found")
    if row.get("data") is None:
        from .. import httpcore

        url = row["source_url"]
        bucket = httpcore.connector_for_url(url)
        client = httpcore.client(bucket)
        try:
            resp = await client.get(url, timeout=20.0)
            resp.raise_for_status()
            data = resp.content
            mime = (resp.headers.get("content-type") or "image/jpeg").split(";")[0]
        except Exception as e:  # noqa: BLE001
            raise ProblemError(502, "upstream_error", f"image fetch failed: {e}") from e
        if len(data) > 20 * 1024 * 1024:
            raise ProblemError(413, "upload_too_large", "image is too large")
        etag = hashlib.sha256(data).hexdigest()
        width: int | None = None
        height: int | None = None
        try:
            with Image.open(io.BytesIO(data)) as im:
                width, height = im.size
        except Exception:  # noqa: BLE001
            pass
        await db.execute(
            """
            UPDATE images
            SET data = :d, mime = :m, etag = :e, width = :w, height = :h, fetched_at = NOW()
            WHERE id = :i
            """,
            {"d": data, "m": mime, "e": etag, "w": width, "h": height, "i": image_id},
        )
        row = {**row, "data": data, "mime": mime, "etag": etag, "width": width, "height": height}
    return row


_touch_tasks: set[asyncio.Task[Any]] = set()
_touching_ids: set[str] = set()


def _schedule_touch_access(image_id: str) -> None:
    """Update last_accessed_at asynchronously so cached covers return without DB write latency."""
    if image_id in _touching_ids:
        return
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return
    _touching_ids.add(image_id)

    async def _run() -> None:
        try:
            await db.execute(
                """
                UPDATE images
                SET last_accessed_at = NOW()
                WHERE id = :i
                  AND (last_accessed_at IS NULL
                       OR last_accessed_at < NOW() - INTERVAL '1 hour')
                """,
                {"i": image_id},
            )
        except Exception:  # noqa: BLE001 — non-critical LRU touch
            pass
        finally:
            _touching_ids.discard(image_id)

    task = loop.create_task(_run())
    _touch_tasks.add(task)
    task.add_done_callback(_touch_tasks.discard)


async def get_image(image_id: str, size: str = "md") -> Response:
    if image_id == PLACEHOLDER_ID:
        data, mime, etag = _placeholder(size)
        return Response(
            content=data,
            media_type=mime,
            headers={
                "Cache-Control": "public, max-age=31536000, immutable",
                "ETag": f'"{etag}"',
            },
        )

    try:
        row = await _load(image_id)
    except ProblemError as pe:
        if pe.status == 502:
            data, mime, etag = _placeholder(size)
            return Response(
                content=data,
                media_type=mime,
                headers={
                    "Cache-Control": "public, max-age=300",
                    "ETag": f'"{etag}"',
                },
            )
        raise

    if size == "original":
        data = row["data"]
        mime = row.get("mime") or "image/jpeg"
        etag = row.get("etag") or hashlib.sha256(data).hexdigest()
    else:
        variant = await db.fetch_one(
            "SELECT data, mime, etag FROM image_variants WHERE image_id = :i AND size = :s",
            {"i": image_id, "s": size},
        )
        if variant:
            data, mime, etag = variant["data"], variant["mime"], variant["etag"]
        else:
            # PIL is CPU-bound: resizing on the event loop stalled every other
            # request (a screenful of 20 new covers serialised the API).
            data, mime, etag = await asyncio.to_thread(_resize, row["data"], size)
            if not mime:
                mime = row.get("mime") or "image/jpeg"
            await db.execute(
                """
                INSERT INTO image_variants (image_id, size, mime, data, etag)
                VALUES (:i, :s, :m, :d, :e)
                ON CONFLICT (image_id, size) DO UPDATE
                SET mime = EXCLUDED.mime, data = EXCLUDED.data, etag = EXCLUDED.etag
                """,
                {"i": image_id, "s": size, "m": mime, "d": data, "e": etag},
            )
    # Sliding TTL for disposable search covers; throttled and non-blocking so cached covers
    # don't wait on a DB write round-trip.
    _schedule_touch_access(image_id)
    return Response(
        content=data,
        media_type=mime,
        headers={
            "Cache-Control": "public, max-age=31536000, immutable",
            "ETag": f'"{etag}"',
        },
    )
