"""Application-level encryption for sensitive memory content.

Provides AES-256-GCM encryption using a machine-derived key. This prevents
casual exfiltration of memory databases — someone who copies your
~/.memory-router/ directory to another machine can't read the content.

The key is derived from a stored random salt + machine identity using PBKDF2.
This is not Fort Knox (an attacker with root on YOUR machine can derive the
same key), but it stops:
  - Casual file-copy exfiltration
  - Backup/cloud-sync exposure (iCloud, Dropbox, etc.)
  - Multi-user systems where another account has read access

Install the `cryptography` package for real AES-256-GCM:
    pip install cryptography

Without it, encryption gracefully degrades to a no-op (plaintext passthrough)
so the CLI still works without the optional dependency.
"""

from __future__ import annotations

import hashlib
import os
from typing import Optional

from ..config import ROOT_DIR, ensure_dirs
from ..utils.fs import atomic_write_bytes


_SALT_FILE = ROOT_DIR / ".encryption_salt"
_KEY_CACHE: Optional[bytes] = None


def _get_machine_id() -> bytes:
    """Best-effort machine identifier. Combines hostname + user."""
    import getpass
    import platform

    parts = [platform.node(), getpass.getuser()]
    return ":".join(parts).encode("utf-8")


def _derive_key() -> bytes:
    """Derive a 256-bit encryption key from machine identity + stored salt.

    The salt is generated once and stored under ~/.memory-router/. It ensures
    the key is unique per installation even if two machines share hostname+user.
    """
    global _KEY_CACHE
    if _KEY_CACHE is not None:
        return _KEY_CACHE

    ensure_dirs()
    if _SALT_FILE.exists():
        salt = _SALT_FILE.read_bytes()
    else:
        salt = os.urandom(32)
        atomic_write_bytes(_SALT_FILE, salt)

    machine_id = _get_machine_id()
    _KEY_CACHE = hashlib.pbkdf2_hmac(
        "sha256", machine_id, salt, iterations=100_000
    )
    return _KEY_CACHE


def is_encryption_available() -> bool:
    """Check if the cryptography package is installed."""
    try:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM  # noqa: F401

        return True
    except ImportError:
        return False


def encrypt_content(plaintext: str, key: Optional[bytes] = None) -> bytes:
    """AES-256-GCM encryption. Returns nonce(12) + ciphertext + tag(16).

    Raises RuntimeError if cryptography is not installed, to prevent
    silent plaintext fallback.
    """
    try:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    except ImportError:
        import warnings
        warnings.warn(
            "cryptography package not installed — encryption is unavailable. "
            "Content will be stored as plaintext. Install with: "
            "pip install memory-router[encryption]",
            stacklevel=2,
        )
        return plaintext.encode("utf-8")

    key = key or _derive_key()
    nonce = os.urandom(12)
    aes = AESGCM(key)
    ct = aes.encrypt(nonce, plaintext.encode("utf-8"), None)
    return nonce + ct


def decrypt_content(data: bytes, key: Optional[bytes] = None) -> str:
    """Decrypt AES-256-GCM content.

    Falls back to UTF-8 decoding if cryptography is not installed.
    """
    try:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    except ImportError:
        return data.decode("utf-8")

    key = key or _derive_key()
    if len(data) < 28:
        # Too short to be a valid nonce + ciphertext + tag — treat as plaintext
        return data.decode("utf-8")

    nonce, ct = data[:12], data[12:]
    aes = AESGCM(key)
    return aes.decrypt(nonce, ct, None).decode("utf-8")


def compute_hmac(data: bytes, key: Optional[bytes] = None) -> bytes:
    """Compute HMAC-SHA256 for integrity verification."""
    import hmac as _hmac

    key = key or _derive_key()
    return _hmac.new(key, data, hashlib.sha256).digest()


def verify_hmac(data: bytes, signature: bytes, key: Optional[bytes] = None) -> bool:
    """Verify HMAC-SHA256 signature."""
    import hmac as _hmac

    key = key or _derive_key()
    expected = _hmac.new(key, data, hashlib.sha256).digest()
    return _hmac.compare_digest(expected, signature)
