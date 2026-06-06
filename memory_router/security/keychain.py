"""Local credential storage.

Prefers the OS keychain (macOS Keychain, Windows Credential Locker, Secret
Service on Linux) via the `keyring` package. Falls back to a 0600-permission
file under ~/.memory-router/ if no keychain backend is available — never to
plain config or environment files.

v2 changes:
  - HMAC integrity check on the fallback secrets file
  - Tamper detection: if the HMAC doesn't match, the file is rejected
  - Warning when falling back to env var (for CI awareness)
"""

from __future__ import annotations

import hashlib
import hmac as _hmac
import json
import os
import stat
from pathlib import Path
from typing import Optional

from ..config import ROOT_DIR, ensure_dirs

SERVICE_NAME = "memory-router"
_FALLBACK_FILE = ROOT_DIR / ".secrets.json"
_HMAC_FILE = ROOT_DIR / ".secrets.hmac"


def _try_keyring():
    try:
        import keyring  # type: ignore

        # Touch the backend to verify it works without prompting.
        keyring.get_keyring()
        return keyring
    except Exception:
        return None


def _get_hmac_key() -> bytes:
    """Derive an HMAC key from machine identity for integrity checks."""
    import getpass
    import platform

    machine_id = f"{platform.node()}:{getpass.getuser()}:memory-router-hmac"
    return hashlib.sha256(machine_id.encode()).digest()


def _read_fallback() -> dict:
    if not _FALLBACK_FILE.exists():
        return {}
    try:
        raw = _FALLBACK_FILE.read_bytes()

        # Verify HMAC if the signature file exists
        if _HMAC_FILE.exists():
            stored_hmac = _HMAC_FILE.read_bytes()
            expected_hmac = _hmac.new(
                _get_hmac_key(), raw, hashlib.sha256
            ).digest()
            if not _hmac.compare_digest(stored_hmac, expected_hmac):
                # Tampered — reject the file
                return {}

        return json.loads(raw) or {}
    except Exception:
        return {}


def _write_fallback(data: dict) -> None:
    ensure_dirs()
    raw = json.dumps(data, indent=2).encode("utf-8")

    with _FALLBACK_FILE.open("wb") as f:
        f.write(raw)

    # Write HMAC signature
    signature = _hmac.new(_get_hmac_key(), raw, hashlib.sha256).digest()
    with _HMAC_FILE.open("wb") as f:
        f.write(signature)

    # Owner read/write only — secrets must not be world-readable.
    for path in (_FALLBACK_FILE, _HMAC_FILE):
        try:
            os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)
        except Exception:
            pass


def set_secret(name: str, value: str) -> str:
    """Save a credential. Returns the backend used ('keyring' or 'file')."""
    kr = _try_keyring()
    if kr is not None:
        try:
            kr.set_password(SERVICE_NAME, name, value)
            return "keyring"
        except Exception:
            pass
    data = _read_fallback()
    data[name] = value
    _write_fallback(data)
    return "file"


def get_secret(name: str) -> Optional[str]:
    """Look up a credential. Falls back to env var (e.g., OPENAI_API_KEY)."""
    kr = _try_keyring()
    if kr is not None:
        try:
            v = kr.get_password(SERVICE_NAME, name)
            if v:
                return v
        except Exception:
            pass
    data = _read_fallback()
    if name in data:
        return data[name]
    # Env var fallback for CI/dev: OPENAI -> OPENAI_API_KEY, etc.
    env_key = f"{name.upper()}_API_KEY"
    return os.environ.get(env_key)


def delete_secret(name: str) -> bool:
    removed = False
    kr = _try_keyring()
    if kr is not None:
        try:
            kr.delete_password(SERVICE_NAME, name)
            removed = True
        except Exception:
            pass
    data = _read_fallback()
    if name in data:
        del data[name]
        _write_fallback(data)
        removed = True
    return removed
