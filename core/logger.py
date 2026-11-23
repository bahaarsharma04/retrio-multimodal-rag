import sys
import time
import logging
from contextlib import contextmanager
from logging.handlers import RotatingFileHandler
from typing import Dict, Any, Optional

from core.config import LOGS_DIR

# Global log format
LOG_FORMAT = "[%(asctime)s] [%(levelname)s] [%(name)s] [%(filename)s:%(lineno)d] - %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# 10 MB per log file, up to 5 rotating backup archives
MAX_LOG_BYTES = 10 * 1024 * 1024
BACKUP_COUNT = 5

_loggers: Dict[str, logging.Logger] = {}


def setup_logger(name: str, log_filename: str, level=logging.INFO) -> logging.Logger:
    """Configures a logger with both dedicated rotating file handler and console stream."""
    if name in _loggers:
        return _loggers[name]

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False  # Avoid duplicate entries in root logger

    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    # 1. Rotating File Handler (UTF-8 safe)
    log_file_path = LOGS_DIR / log_filename
    file_handler = RotatingFileHandler(
        log_file_path,
        maxBytes=MAX_LOG_BYTES,
        backupCount=BACKUP_COUNT,
        encoding="utf-8",
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # 2. Console Stream Handler (Standard output)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    _loggers[name] = logger
    return logger


# Dedicated specialized loggers
app_logger = setup_logger("retrio.app", "app.log")
ingest_logger = setup_logger("retrio.ingest", "ingestion.log")
retrieval_logger = setup_logger("retrio.retrieval", "retrieval.log")
generation_logger = setup_logger("retrio.generation", "generation.log")
chunking_logger = setup_logger("retrio.chunking", "chunking.log")
embedding_logger = setup_logger("retrio.embedding", "embedding.log")


@contextmanager
def log_timed_block(logger: logging.Logger, action_name: str, meta: Optional[Dict[str, Any]] = None):
    """Context manager to measure and log execution duration automatically."""
    t0 = time.perf_counter()
    meta_str = f" | {meta}" if meta else ""
    logger.info(f"▶ STARTING: {action_name}{meta_str}")
    try:
        yield
    except Exception as e:
        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.error(f"❌ FAILED: {action_name} after {elapsed_ms:.1f}ms - Error: {e}", exc_info=True)
        raise
    else:
        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.info(f"✔ FINISHED: {action_name} in {elapsed_ms:.1f}ms")
