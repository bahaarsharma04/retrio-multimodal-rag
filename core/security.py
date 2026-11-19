import os
import bcrypt
import jwt
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Any

from core.config import JWT_SECRET_KEY, JWT_ALGORITHM, ACCESS_TOKEN_EXPIRE_MINUTES
from core.logger import app_logger


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
    Creates a signed JWT access token.
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
    })
    encoded_jwt = jwt.encode(to_encode, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
    return encoded_jwt


def decode_access_token(token: str) -> Optional[Dict[str, Any]]:
    """
    Decodes and validates a JWT access token.
    Returns the decoded claims dictionary, or None if invalid/expired.
    """
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        app_logger.info("Provided JWT token has expired.")
        return None
    except jwt.InvalidTokenError as e:
        app_logger.warning(f"Invalid JWT token: {e}")
        return None
