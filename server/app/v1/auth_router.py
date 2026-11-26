"""KWhale v1 auth routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request, Response

from .. import db
from ..auth import (
    LoginRequest,
    LogoutRequest,
    RefreshRequest,
    RegisterRequest,
    check_login_throttle,
    clear_login_failures,
    current_user,
    login,
    logout_session,
    public_user,
    record_login_failure,
    refresh_session,
    register,
)
from ..errors import ProblemError
from .schemas import AuthResponse, User, UserUpdate

router = APIRouter(prefix="/auth", tags=["auth"])
me_router = APIRouter(prefix="/me", tags=["me"])


@router.post("/register", response_model=AuthResponse, status_code=201)
async def register_endpoint(body: RegisterRequest) -> dict:
    return await register(body)


@router.post("/login", response_model=AuthResponse)
async def login_endpoint(body: LoginRequest, request: Request) -> dict:
    key = f"{request.client.host if request.client else '?'}:{body.email.strip().lower()}"
    check_login_throttle(key)
    try:
        result = await login(body)
    except ProblemError:
        record_login_failure(key)
        raise
    clear_login_failures(key)
    return result


@router.post("/refresh", response_model=AuthResponse)
async def refresh_endpoint(body: RefreshRequest) -> dict:
    return await refresh_session(body)


@router.post("/logout", status_code=204)
async def logout_endpoint(body: LogoutRequest) -> Response:
    await logout_session(body)
    return Response(status_code=204)


@me_router.get("", response_model=User)
async def get_me(user: dict = Depends(current_user)) -> User:
    return User(**public_user(user))


@me_router.patch("", response_model=User)
async def patch_me(body: UserUpdate, user: dict = Depends(current_user)) -> User:
    if body.display_name is not None:
        name = body.display_name.strip()
        if not name:
            raise ProblemError(400, "validation_failed", "displayName must not be empty")
        await db.execute(
            "UPDATE users SET display_name = :n WHERE id = :i",
            {"n": name, "i": user["id"]},
        )
        user = {**user, "display_name": name}
    return User(**public_user(user))
