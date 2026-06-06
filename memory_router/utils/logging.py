"""Structured logging for Memory Router.

Provides a single get_logger() entry point that returns a stdlib logger
configured with JSON-structured output to ~/.memory-router/logs/. Logs
are rotated at 5 MB with 3 backups. Console output stays human-readable.

Usage:
    from memory_router.utils.logging import get_logger
    log = get_logger(__name__)
    log.info("memory stored", extra={"memory_id": 42, "domain": "code"})
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional


class _JSONFormatter(logging.Formatter):
    """Formats log records as single-line JSON for machine consumption."""

    def format(self, record: logging.LogRecord) -> str:
        entry: Dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        # Merge any extra keys the caller passed
        for key in ("memory_id", "domain", "task", "provider", "model",
                     "latency_ms", "tokens", "error", "session_id",
                     "action", "count", "detail"):
            val = getattr(record, key, None)
            if val is not None:
                entry[key] = val
        if record.exc_info and record.exc_info[1]:
            entry["exception"] = str(record.exc_info[1])
        return json.dumps(entry, ensure_ascii=False)


_CONFIGURED = False


def get_logger(name: str, log_dir: Optional[Path] = None) -> logging.Logger:
    """Return a logger with JSON file + human-readable console handlers.

    Safe to call repeatedly — handlers are attached once.
    """
    global _CONFIGURED
    logger = logging.getLogger(name)

    if _CONFIGURED:
        return logger

    from ..config import LOG_DIR

    log_dir = log_dir or LOG_DIR
    log_dir.mkdir(parents=True, exist_ok=True)

    level_str = os.environ.get("MEMORY_ROUTER_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_str, logging.INFO)
    logger = logging.getLogger("memory_router")
    logger.setLevel(level)

    if not logger.handlers:
        # JSON file handler with rotation
        try:
            from logging.handlers import RotatingFileHandler
            fh = RotatingFileHandler(
                str(log_dir / "memory-router.log"),
                maxBytes=5 * 1024 * 1024,  # 5 MB
                backupCount=3,
                encoding="utf-8",
            )
            fh.setFormatter(_JSONFormatter())
            fh.setLevel(level)
            logger.addHandler(fh)
        except Exception:
            pass  # Don't break if log dir is unwritable

        # Console handler — only for warnings+
        ch = logging.StreamHandler()
        ch.setLevel(logging.WARNING)
        ch.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
        logger.addHandler(ch)

    _CONFIGURED = True
    return logging.getLogger(name)
