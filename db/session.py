import os
import time
from contextlib import contextmanager
from typing import Generator
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from core.config import DATABASE_URL
from db.base import Base

# Engine configuration
connect_args = {}
if DATABASE_URL.startswith("sqlite"):
    connect_args["check_same_thread"] = False
elif "postgresql" in DATABASE_URL:
    # 2 second connection timeout prevents UI freezes when Postgres server is unreachable
    connect_args["connect_timeout"] = 2

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    connect_args=connect_args,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Status cache to avoid continuous connection polling when DB is offline
_DB_STATUS_CACHE = {"status": None, "last_checked": 0.0}
_CACHE_TTL_SECONDS = 15.0


def init_db() -> bool:
    """Ensures database tables are created."""
    try:
        from db import models  # noqa: F401
        Base.metadata.create_all(bind=engine)
        return True
    except Exception as e:
        print(f"Warning: Database init failed: {e}")
        return False


def is_db_connected(force_check: bool = False) -> bool:
    """
    Probes whether PostgreSQL or the configured database is reachable.
    Caches result for 15s to keep Streamlit UI fast when offline.
    """
    now = time.time()
    if not force_check and _DB_STATUS_CACHE["status"] is not None:
        if now - _DB_STATUS_CACHE["last_checked"] < _CACHE_TTL_SECONDS:
            return _DB_STATUS_CACHE["status"]

    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
            _DB_STATUS_CACHE["status"] = True
            _DB_STATUS_CACHE["last_checked"] = now
            return True
    except Exception:
        _DB_STATUS_CACHE["status"] = False
        _DB_STATUS_CACHE["last_checked"] = now
        return False


def get_db() -> Generator[Session, None, None]:
    """FastAPI Dependency for database sessions."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@contextmanager
def get_db_session() -> Generator[Session, None, None]:
    """Context manager for non-FastAPI standalone operations."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
