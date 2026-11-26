"""RFC 7807 problem responses for the public KWhale API."""

from __future__ import annotations

from typing import Any

from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

_TITLES = {
    400: "Bad Request",
    401: "Unauthorized",
    403: "Forbidden",
    404: "Not Found",
    409: "Conflict",
    413: "Payload Too Large",
    422: "Unprocessable Entity",
    429: "Too Many Requests",
    500: "Internal Server Error",
    501: "Not Implemented",
    502: "Bad Gateway",
}


class ProblemError(Exception):
    def __init__(
        self,
        status: int,
        code: str,
        detail: str,
        *,
        title: str | None = None,
        type_uri: str | None = None,
        fields: dict[str, str] | None = None,
        instance: str | None = None,
    ) -> None:
        self.status = status
        self.code = code
        self.detail = detail
        self.title = title or _TITLES.get(status, "Error")
        self.type_uri = type_uri or f"https://api.kwhale.example/problems/{code}"
        self.fields = fields or {}
        self.instance = instance
        super().__init__(detail)


def problem_body(
    status: int,
    code: str,
    detail: str,
    *,
    title: str | None = None,
    type_uri: str | None = None,
    fields: dict[str, str] | None = None,
    instance: str | None = None,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "type": type_uri or f"https://api.kwhale.example/problems/{code}",
        "title": title or _TITLES.get(status, "Error"),
        "status": status,
        "detail": detail,
        "code": code,
    }
    if instance:
        body["instance"] = instance
    if fields:
        body["fields"] = fields
    return body


def _response(body: dict[str, Any], status: int) -> JSONResponse:
    return JSONResponse(status_code=status, content=body, media_type="application/problem+json")


async def problem_handler(request: Request, exc: ProblemError) -> JSONResponse:
    return _response(
        problem_body(
            exc.status,
            exc.code,
            exc.detail,
            title=exc.title,
            type_uri=exc.type_uri,
            fields=exc.fields,
            instance=exc.instance or request.url.path,
        ),
        exc.status,
    )


async def http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    detail = exc.detail if isinstance(exc.detail, str) else "request failed"
    code = {
        400: "validation_failed",
        401: "invalid_credentials",
        403: "forbidden",
        404: "not_found",
        409: "conflict",
        413: "upload_too_large",
        429: "rate_limited",
        501: "not_implemented",
        502: "upstream_error",
    }.get(exc.status_code, "error")
    return _response(
        problem_body(
            exc.status_code,
            code,
            detail,
            fields=None,
            instance=request.url.path,
        ),
        exc.status_code,
    )


async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    fields: dict[str, str] = {}
    for err in exc.errors():
        loc = ".".join(str(part) for part in err.get("loc", ()) if part != "body")
        fields[loc or "body"] = str(err.get("msg", "invalid value"))
    return _response(
        problem_body(
            422,
            "validation_failed",
            "Request validation failed",
            fields=fields,
            instance=request.url.path,
        ),
        422,
    )
