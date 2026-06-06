"""Local credential storage.

Prefers the OS keychain (macOS Keychain, Windows Credential Locker, Secret
Service on Linux) via the `keyring` package. Falls back to an encrypted file
under ~/.memory-router/ when no keychain backend is available.

The fallback file is machine-bound, AES-GCM encrypted, and integrity checked.
If the optional `cryptography` dependency is unavailable, the fallback path
raises instead of silently storing plaintext secrets.
"""

from __future__ import annotations

import base64
import getpass
import hashlib
import hmac as _hmac
import json
import os
import platform
import stat
from pathlib import Path
from typing import Optional

from ..config import ROOT_DIR, ensure_dirs
from ..utils.fs import atomic_write_bytes

SERVICE_NAME = "memory-router"
_FALLBACK_FILE = ROOT_DIR / ".secrets.json"
_HMAC_FILE = ROOT_DIR / ".secrets.hmac"
_FALLBACK_VERSION = 2


def _try_keyring():
    try:
        import keyring  # type: ignore

        # Touch the backend to verify it works without prompting.
        keyring.get_keyring()
        return keyring
    except Exception:
        return None


def _get_machine_id() -> bytes:
    """Best-effort machine identifier. Combines hostname + user."""
    machine_id = f"{platform.node()}:{getpass.getuser()}:memory-router-secret"
    return machine_id.encode("utf-8")


def _derive_secret_key(salt: bytes) -> bytes:
    """Derive an AES key bound to this machine + installation salt."""
    return hashlib.pbkdf2_hmac(
        "sha256",
        _get_machine_id(),
        salt,
        iterations=100_000,
        dklen=32,
    )


def _get_hmac_key() -> bytes:
    """Derive an HMAC key for integrity checks on the fallback payload."""
    machine_id = f"{platform.node()}:{getpass.getuser()}:memory-router-hmac"
    return hashlib.sha256(machine_id.encode()).digest()


def _encode_b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("ascii")


def _decode_b64(value: str) -> bytes:
    return base64.urlsafe_b64decode(value.encode("ascii"))


def _encrypt_secret_payload(data: dict) -> bytes:
    try:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    except ImportError as e:
        raise RuntimeError(
            "Encrypted fallback secrets require the cryptography package. "
            "Install: pip install memory-router[encryption]"
        ) from e

    salt = os.urandom(16)
    nonce = os.urandom(12)
    key = _derive_secret_key(salt)
    aes = AESGCM(key)
    raw = json.dumps(data, indent=2, sort_keys=True).encode("utf-8")
    ciphertext = aes.encrypt(nonce, raw, None)
    payload = {
        "version": _FALLBACK_VERSION,
        "salt": _encode_b64(salt),
        "nonce": _encode_b64(nonce),
        "ciphertext": _encode_b64(ciphertext),
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _decrypt_secret_payload(payload: dict) -> dict:
    try:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    except ImportError as e:
        raise RuntimeError(
            "Encrypted fallback secrets require the cryptography package. "
            "Install: pip install memory-router[encryption]"
        ) from e

    salt = _decode_b64(payload["salt"])
    nonce = _decode_b64(payload["nonce"])
    ciphertext = _decode_b64(payload["ciphertext"])
    key = _derive_secret_key(salt)
    raw = AESGCM(key).decrypt(nonce, ciphertext, None)
    return json.loads(raw.decode("utf-8")) or {}


def _read_fallback() -> dict:
    if not _FALLBACK_FILE.exists():
        return {}
    try:
        raw = _FALLBACK_FILE.read_bytes()

        # Verify HMAC if the signature file exists.
        if _HMAC_FILE.exists():
            stored_hmac = _HMAC_FILE.read_bytes()
            expected_hmac = _hmac.new(
                _get_hmac_key(), raw, hashlib.sha256
            ).digest()
            if not _hmac.compare_digest(stored_hmac, expected_hmac):
                return {}

        payload = json.loads(raw.decode("utf-8"))
        if isinstance(payload, dict) and payload.get("version") == _FALLBACK_VERSION:
            if {"salt", "nonce", "ciphertext"} <= payload.keys():
                return _decrypt_secret_payload(payload)
            return {}

        if isinstance(payload, dict):
            # Legacy plaintext fallback support.
            return payload
        return {}
    except RuntimeError:
        raise
    except Exception:
        return {}


def _write_fallback(data: dict) -> None:
    ensure_dirs()
    payload = _encrypt_secret_payload(data)
    signature = _hmac.new(_get_hmac_key(), payload, hashlib.sha256).digest()

    atomic_write_bytes(_FALLBACK_FILE, payload)
    atomic_write_bytes(_HMAC_FILE, signature)

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
    return "encrypted-file"


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
