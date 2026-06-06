"""Helpers for managing a local Ollama server."""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from urllib.parse import urlparse

from ..providers.ollama_provider import OllamaProvider


_LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}


def _normalize_host(host: str) -> str:
    raw = (host or "").strip()
    if not raw:
        return "localhost:11434"
    if "://" not in raw:
        raw = f"http://{raw}"
    parsed = urlparse(raw)
    return parsed.netloc or parsed.path


def _is_local_host(host: str) -> bool:
    normalized = _normalize_host(host)
    hostname = normalized.split(":", 1)[0].strip("[]").lower()
    return hostname in _LOCAL_HOSTS


def ensure_ollama_running(host: str = "http://localhost:11434", timeout: int = 15) -> bool:
    """Ensure the Ollama server is reachable, starting it in the background if needed.

    Returns True when the server is available. Raises RuntimeError when Ollama
    cannot be started or does not become ready within the timeout.
    """
    provider = OllamaProvider(host=host)
    if provider.is_available():
        return True

    # Give a briefly-starting process a moment before we try to launch another one.
    time.sleep(0.5)
    if provider.is_available():
        return True

    if not _is_local_host(host):
        raise RuntimeError(
            f"Auto-start only works for local Ollama hosts; configured host is {host!r}."
        )

    if shutil.which("ollama") is None:
        raise RuntimeError(
            "Ollama is not installed or not on PATH. Install it from https://ollama.com "
            "or run `brew install ollama` on macOS."
        )

    env = os.environ.copy()
    env["OLLAMA_HOST"] = _normalize_host(host)

    popen_kwargs = {
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
        "env": env,
    }
    if os.name == "nt":
        popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        popen_kwargs["start_new_session"] = True

    proc = subprocess.Popen(["ollama", "serve"], **popen_kwargs)

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if provider.is_available():
            return True
        if proc.poll() is not None:
            break
        time.sleep(0.5)

    if provider.is_available():
        return True

    if proc.poll() is not None:
        raise RuntimeError(
            "Ollama exited before it became ready. Try running `ollama serve` "
            "manually to inspect the error output."
        )

    raise RuntimeError(
        f"Ollama did not become ready on {host!r} within {timeout} seconds. "
        "Try running `ollama serve` manually."
    )
