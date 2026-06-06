"""Tests for credential storage with HMAC integrity."""

import json
from pathlib import Path
from unittest.mock import patch

from memory_router.security.keychain import (
    _read_fallback,
    _write_fallback,
    set_secret,
    get_secret,
    delete_secret,
)


def test_fallback_write_and_read(tmp_path, monkeypatch):
    """Fallback file should round-trip with HMAC verification."""
    fb_file = tmp_path / ".secrets.json"
    hmac_file = tmp_path / ".secrets.hmac"
    monkeypatch.setattr("memory_router.security.keychain._FALLBACK_FILE", fb_file)
    monkeypatch.setattr("memory_router.security.keychain._HMAC_FILE", hmac_file)

    _write_fallback({"openai": "sk-test123"})
    assert fb_file.exists()
    assert hmac_file.exists()

    data = _read_fallback()
    assert data["openai"] == "sk-test123"
    assert b"sk-test123" not in fb_file.read_bytes()


def test_tampered_fallback_rejected(tmp_path, monkeypatch):
    """Tampered secrets file should be rejected."""
    fb_file = tmp_path / ".secrets.json"
    hmac_file = tmp_path / ".secrets.hmac"
    monkeypatch.setattr("memory_router.security.keychain._FALLBACK_FILE", fb_file)
    monkeypatch.setattr("memory_router.security.keychain._HMAC_FILE", hmac_file)

    _write_fallback({"openai": "sk-real"})

    # Tamper with the file
    fb_file.write_text(json.dumps({"openai": "sk-evil"}), encoding="utf-8")

    data = _read_fallback()
    assert data == {}  # Rejected due to HMAC mismatch


def test_missing_hmac_file_still_reads(tmp_path, monkeypatch):
    """When no HMAC file exists, fallback should still read (legacy compat)."""
    fb_file = tmp_path / ".secrets.json"
    hmac_file = tmp_path / ".secrets.hmac"
    monkeypatch.setattr("memory_router.security.keychain._FALLBACK_FILE", fb_file)
    monkeypatch.setattr("memory_router.security.keychain._HMAC_FILE", hmac_file)

    # Write directly without HMAC (simulating legacy)
    fb_file.write_text(json.dumps({"openai": "sk-legacy"}), encoding="utf-8")

    data = _read_fallback()
    assert data["openai"] == "sk-legacy"


def test_env_var_fallback(tmp_path, monkeypatch):
    """Should fall back to env var when no keyring or file secret exists."""
    fb_file = tmp_path / ".secrets.json"
    hmac_file = tmp_path / ".secrets.hmac"
    monkeypatch.setattr("memory_router.security.keychain._FALLBACK_FILE", fb_file)
    monkeypatch.setattr("memory_router.security.keychain._HMAC_FILE", hmac_file)
    monkeypatch.setattr("memory_router.security.keychain._try_keyring", lambda: None)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env")

    val = get_secret("openai")
    assert val == "sk-from-env"


def test_delete_secret(tmp_path, monkeypatch):
    """Deleting a secret should remove it from the fallback file."""
    fb_file = tmp_path / ".secrets.json"
    hmac_file = tmp_path / ".secrets.hmac"
    monkeypatch.setattr("memory_router.security.keychain._FALLBACK_FILE", fb_file)
    monkeypatch.setattr("memory_router.security.keychain._HMAC_FILE", hmac_file)
    monkeypatch.setattr("memory_router.security.keychain._try_keyring", lambda: None)

    _write_fallback({"openai": "sk-test", "anthropic": "sk-ant"})
    ok = delete_secret("openai")
    assert ok is True

    data = _read_fallback()
    assert "openai" not in data
    assert "anthropic" in data


def test_set_secret_uses_encrypted_file_when_keyring_unavailable(tmp_path, monkeypatch):
    fb_file = tmp_path / ".secrets.json"
    hmac_file = tmp_path / ".secrets.hmac"
    monkeypatch.setattr("memory_router.security.keychain._FALLBACK_FILE", fb_file)
    monkeypatch.setattr("memory_router.security.keychain._HMAC_FILE", hmac_file)
    monkeypatch.setattr("memory_router.security.keychain._try_keyring", lambda: None)

    backend = set_secret("openai", "sk-super-secret")
    assert backend == "encrypted-file"
    assert b"sk-super-secret" not in fb_file.read_bytes()
    assert get_secret("openai") == "sk-super-secret"
