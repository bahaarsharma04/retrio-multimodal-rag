from fastapi import APIRouter, HTTPException, Request, Depends, status
from api.schemas import (
    UserRegisterRequest,
    UserLoginRequest,
    UserResponse,
    TokenResponse,
)
from core.users import create_user, authenticate_user, get_user_by_email
from core.security import create_access_token
from core.limiter import limiter
from api.deps import get_current_active_user

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

    token = create_access_token({
        "sub": user["id"],
        "email": user["email"],
        "role": user["role"],
    })

    return TokenResponse(
        access_token=token,
        token_type="bearer",
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

    token = create_access_token({
        "sub": user["id"],
        "email": user["email"],
        "role": user["role"],
    })

    return TokenResponse(
        access_token=token,
        token_type="bearer",
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
