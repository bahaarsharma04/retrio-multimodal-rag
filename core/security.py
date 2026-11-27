import uuid
import bcrypt
import jwt
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Any, Set

from core.config import (
    JWT_SECRET_KEY,
    JWT_ALGORITHM,
    ACCESS_TOKEN_EXPIRE_MINUTES,
    REFRESH_TOKEN_EXPIRE_DAYS,
)
from core.logger import app_logger

# Thread-safe in-memory token revocation blocklist
_REVOKED_TOKENS: Set[str] = set()


def revoke_token(token: str) -> None:
    """Adds a JWT token string to the revocation blocklist."""
    if token:
        _REVOKED_TOKENS.add(token.strip())


def is_token_revoked(token: str) -> bool:
    """Checks whether a token has been explicitly revoked via logout."""
    if not token:
        return True
    return token.strip() in _REVOKED_TOKENS


def hash_password(password: str) -> str:
    """Hashes a plaintext password using bcrypt with a generated salt."""
    # Bcrypt supports up to 72 bytes; slice safely if needed
    pwd_bytes = password.encode("utf-8")[:72]
    salt = bcrypt.gensalt(rounds=12)
    return bcrypt.hashpw(pwd_bytes, salt).decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verifies a plaintext password against a stored bcrypt hash."""
    try:
        pwd_bytes = plain_password.encode("utf-8")[:72]
        hash_bytes = hashed_password.encode("utf-8")
        return bcrypt.checkpw(pwd_bytes, hash_bytes)
    except Exception as e:
        app_logger.warning(f"Password verification failure: {e}")
        return False


def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    """
    Creates a signed JWT access token (15-30 minute short-lived token).
    'data' dictionary should include 'sub' (the user_id) and optional claims (e.g. 'email', 'role').
    """
    to_encode = data.copy()
    now = datetime.now(timezone.utc)
    if expires_delta:
        expire = now + expires_delta
    else:
        expire = now + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)

    to_encode.update({
        "exp": expire,
        "iat": now,
        "type": "access",
        "jti": uuid.uuid4().hex,
    })
    encoded_jwt = jwt.encode(to_encode, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
    return encoded_jwt


def create_refresh_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    """
    Creates a signed JWT refresh token (7-day long-lived rotation token).
    """
    to_encode = data.copy()
    now = datetime.now(timezone.utc)
    if expires_delta:
        expire = now + expires_delta
    else:
        expire = now + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)

    to_encode.update({
        "exp": expire,
        "iat": now,
        "type": "refresh",
        "jti": uuid.uuid4().hex,
    })
    encoded_jwt = jwt.encode(to_encode, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
    return encoded_jwt


def decode_access_token(token: str) -> Optional[Dict[str, Any]]:
    """
    Decodes and validates a JWT access token.
    Returns the decoded claims dictionary, or None if invalid/expired/revoked.
    """
    if is_token_revoked(token):
        app_logger.info("Attempted use of revoked access token.")
        return None
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        app_logger.info("Provided JWT token has expired.")
        return None
    except jwt.InvalidTokenError as e:
        app_logger.warning(f"Invalid JWT token: {e}")
        return None


def decode_refresh_token(token: str) -> Optional[Dict[str, Any]]:
    """
    Decodes and validates a JWT refresh token.
    Returns the decoded claims dictionary, or None if invalid/expired/revoked.
    """
    if is_token_revoked(token):
        app_logger.info("Attempted use of revoked refresh token.")
        return None
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        if payload.get("type") != "refresh":
            app_logger.warning("Token type mismatch: provided token is not a refresh token.")
            return None
        return payload
    except jwt.ExpiredSignatureError:
        app_logger.info("Provided refresh token has expired.")
        return None
    except jwt.InvalidTokenError as e:
        app_logger.warning(f"Invalid refresh token: {e}")
        return None
