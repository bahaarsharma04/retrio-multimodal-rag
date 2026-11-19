from slowapi import Limiter
from slowapi.util import get_remote_address
from starlette.requests import Request
from typing import Optional


def user_or_ip_key_func(request: Request) -> str:
    """
    Identifies the rate limiting subject:
    - If a valid JWT Bearer token is provided, limits are tracked per authenticated user ('user:<user_id>').
    - If unauthenticated (e.g. login/register), limits are tracked per remote IP address ('ip:<remote_ip>').
    """
    auth = request.headers.get("Authorization")
    if auth and auth.startswith("Bearer "):
        parts = auth.split(" ")
        if len(parts) == 2:
            token = parts[1]
            try:
                from core.security import decode_access_token
                payload = decode_access_token(token)
                if payload and "sub" in payload:
                    return f"user:{payload['sub']}"
            except Exception:
                pass

    return f"ip:{get_remote_address(request)}"


limiter = Limiter(
    key_func=user_or_ip_key_func,
    default_limits=["120/minute"],
    headers_enabled=False,
)
