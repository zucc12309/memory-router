"""Tests for encryption module."""

from memory_router.security.encryption import (
    encrypt_content,
    decrypt_content,
    compute_hmac,
    verify_hmac,
)


def test_roundtrip_without_cryptography():
    """Even without cryptography installed, encrypt/decrypt should pass through."""
    text = "Hello, world!"
    encrypted = encrypt_content(text)
    decrypted = decrypt_content(encrypted)
    # If cryptography is installed, it's a real roundtrip.
    # If not, it's plaintext passthrough. Either way, we get the original back.
    assert decrypted == text


def test_hmac_roundtrip():
    data = b"some secret data"
    key = b"test-key-32-bytes-long-padding!!"
    sig = compute_hmac(data, key=key)
    assert verify_hmac(data, sig, key=key)
    # Tampered data should fail
    assert not verify_hmac(b"tampered data", sig, key=key)


def test_encrypt_decrypt_with_key():
    text = "sensitive memory content"
    key = b"\x00" * 32  # 256-bit key
    encrypted = encrypt_content(text, key=key)
    decrypted = decrypt_content(encrypted, key=key)
    assert decrypted == text


def test_encrypt_empty_string():
    text = ""
    encrypted = encrypt_content(text)
    decrypted = decrypt_content(encrypted)
    assert decrypted == text
