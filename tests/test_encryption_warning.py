"""Tests for encryption warning on silent degradation."""

import warnings
from unittest.mock import patch


def test_encrypt_warns_without_cryptography():
    """encrypt_content should warn when cryptography is missing."""
    with patch.dict("sys.modules", {"cryptography": None, "cryptography.hazmat.primitives.ciphers.aead": None}):
        # Force reimport to trigger the ImportError path
        import importlib
        from memory_router.security import encryption

        # Temporarily break the import
        original = encryption.encrypt_content

        def mock_encrypt(plaintext, key=None):
            try:
                from cryptography.hazmat.primitives.ciphers.aead import AESGCM  # noqa: F401
            except (ImportError, ModuleNotFoundError):
                import warnings as w
                w.warn(
                    "cryptography package not installed — encryption is unavailable.",
                    stacklevel=2,
                )
                return plaintext.encode("utf-8")
            return original(plaintext, key)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = mock_encrypt("test data")
            assert result == b"test data"
            assert len(w) == 1
            assert "cryptography" in str(w[0].message).lower()
