from typing import Optional
from fastapi import APIRouter, HTTPException, Request, Depends, status
from api.schemas import (
    UserRegisterRequest,
    UserLoginRequest,
    UserResponse,
    TokenResponse,
    RefreshTokenRequest,
    TokenRefreshResponse,
    LogoutRequest,
    LogoutResponse,
)
from core.users import create_user, authenticate_user, get_user_by_email, get_user_by_id
from core.security import (
    create_access_token,
    create_refresh_token,
    decode_refresh_token,
    revoke_token,
)
from core.config import ACCESS_TOKEN_EXPIRE_MINUTES
from core.limiter import limiter
from api.deps import get_current_active_user, oauth2_scheme

router = APIRouter(prefix="/api/v1/auth", tags=["Authentication & Access"])


@router.post(
    "/register",
    response_model=TokenResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register a new Retrio user",
)
@limiter.limit("10/minute")
def register_user(request: Request, payload: UserRegisterRequest):
    """
    Registers a new user with email, password, and optional full name.
    Rate limited to 10 attempts per minute per IP to mitigate automated spam registrations.
    """
    email_clean = payload.email.strip().lower()
    if "@" not in email_clean or "." not in email_clean:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid email address format.",
        )

    if len(payload.password) < 6:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Password must be at least 6 characters long.",
        )

    existing = get_user_by_email(email_clean)
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"An account with email '{email_clean}' already exists.",
        )

    try:
        user = create_user(
            email=email_clean,
            password=payload.password,
            full_name=payload.full_name or "",
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )

    claims = {
        "sub": user["id"],
        "email": user["email"],
        "role": user["role"],
    }
    access_token = create_access_token(claims)
    refresh_token = create_refresh_token(claims)

    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer",
        expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        user=UserResponse(
            id=user["id"],
            email=user["email"],
            full_name=user.get("full_name", ""),
            role=user.get("role", "user"),
            is_active=user.get("is_active", True),
            created_at=user.get("created_at", ""),
        ),
    )


@router.post(
    "/login",
    response_model=TokenResponse,
    summary="Authenticate and obtain JWT access token",
)
@limiter.limit("5/minute")
def login_user(request: Request, payload: UserLoginRequest):
    """
    Authenticates a user using email and password.
    Rate limited to 5 attempts per minute per IP to prevent brute-force attacks.
    """
    user = authenticate_user(payload.email, payload.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    claims = {
        "sub": user["id"],
        "email": user["email"],
        "role": user["role"],
    }
    access_token = create_access_token(claims)
    refresh_token = create_refresh_token(claims)

    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer",
        expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        user=UserResponse(
            id=user["id"],
            email=user["email"],
            full_name=user.get("full_name", ""),
            role=user.get("role", "user"),
            is_active=user.get("is_active", True),
            created_at=user.get("created_at", ""),
        ),
    )


@router.get(
    "/me",
    response_model=UserResponse,
    summary="Get current authenticated user profile",
)
@limiter.limit("60/minute")
def get_current_user_profile(request: Request, current_user: dict = Depends(get_current_active_user)):
    """Returns the authenticated user's profile and active role."""
    return UserResponse(
        id=current_user["id"],
        email=current_user["email"],
        full_name=current_user.get("full_name", ""),
        role=current_user.get("role", "user"),
        is_active=current_user.get("is_active", True),
        created_at=current_user.get("created_at", ""),
    )


@router.post(
    "/refresh",
    response_model=TokenRefreshResponse,
    summary="Obtain a refreshed JWT access token using a valid refresh token",
)
@limiter.limit("20/minute")
def refresh_access_token(request: Request, payload: RefreshTokenRequest):
    """
    Validates the supplied refresh token and issues a new short-lived access token.
    Enforces token expiration and revocation checks.
    """
    claims = decode_refresh_token(payload.refresh_token)
    if not claims or "sub" not in claims:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid, expired, or revoked refresh token. Please log in again.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user_id = claims["sub"]
    user = get_user_by_id(user_id)
    if not user or not user.get("is_active", True):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User account no longer active.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    new_access_token = create_access_token({
        "sub": user["id"],
        "email": user["email"],
        "role": user["role"],
    })

    return TokenRefreshResponse(
        access_token=new_access_token,
        token_type="bearer",
        expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
    )


@router.post(
    "/logout",
    response_model=LogoutResponse,
    summary="Revoke access token and refresh token",
)
def logout_user(
    request: Request,
    payload: Optional[LogoutRequest] = None,
    token: Optional[str] = Depends(oauth2_scheme),
):
    """
    Revokes the current Bearer access token and optional refresh token,
    immediately invalidating them across all authenticated endpoints.
    """
    if token:
        revoke_token(token)

    if payload and payload.refresh_token:
        revoke_token(payload.refresh_token)

    return LogoutResponse(message="Successfully logged out. Tokens have been revoked.")
