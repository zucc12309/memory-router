"""Filesystem helpers for secure local writes.

These utilities keep local config, secrets, and database files locked down
from the moment they are created rather than chmod'ing them after the fact.
"""

from __future__ import annotations

import os
import stat
import tempfile
from pathlib import Path


DEFAULT_SECURE_MODE = stat.S_IRUSR | stat.S_IWUSR


def ensure_secure_file(path: Path, mode: int = DEFAULT_SECURE_MODE) -> None:
    """Create a file with secure permissions if it doesn't exist yet."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        try:
            os.chmod(path, mode)
        except Exception:
            pass
        return

    fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, mode)
    try:
        os.close(fd)
    finally:
        try:
            os.chmod(path, mode)
        except Exception:
            pass


def atomic_write_bytes(path: Path, data: bytes, mode: int = DEFAULT_SECURE_MODE) -> None:
    """Atomically write bytes with secure permissions."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "wb") as f:
            try:
                os.fchmod(f.fileno(), mode)
            except Exception:
                pass
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name, path)
        try:
            os.chmod(path, mode)
        except Exception:
            pass
    finally:
        try:
            if os.path.exists(tmp_name):
                os.unlink(tmp_name)
        except Exception:
            pass


def atomic_write_text(
    path: Path,
    text: str,
    encoding: str = "utf-8",
    mode: int = DEFAULT_SECURE_MODE,
) -> None:
    """Atomically write text with secure permissions."""
    atomic_write_bytes(path, text.encode(encoding), mode=mode)
