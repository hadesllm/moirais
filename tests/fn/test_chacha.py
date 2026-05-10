"""Tests for moirais.crypto._chacha — ChaCha20-Poly1305 AEAD."""

import os

import pytest

from moirais.crypto._chacha import chacha20_poly1305_decrypt, chacha20_poly1305_encrypt


class TestChaCha20Poly1305:

    def test_roundtrip(self):
        key = os.urandom(32)
        nonce = os.urandom(12)
        plaintext = b"Without music, life would be a mistake. -- Friedrich Nietzsche"
        ct, tag = chacha20_poly1305_encrypt(key, nonce, plaintext)
        recovered = chacha20_poly1305_decrypt(key, nonce, ct, tag)
        assert recovered == plaintext

    def test_empty_plaintext(self):
        key = os.urandom(32)
        nonce = os.urandom(12)
        ct, tag = chacha20_poly1305_encrypt(key, nonce, b"")
        recovered = chacha20_poly1305_decrypt(key, nonce, ct, tag)
        assert recovered == b""

    def test_tampered_ciphertext_raises(self):
        key = os.urandom(32)
        nonce = os.urandom(12)
        ct, tag = chacha20_poly1305_encrypt(key, nonce, b"secret data")
        tampered = bytearray(ct)
        tampered[0] ^= 0xFF
        with pytest.raises(ValueError, match="tag mismatch"):
            chacha20_poly1305_decrypt(key, nonce, bytes(tampered), tag)

    def test_tampered_aad_raises(self):
        key = os.urandom(32)
        nonce = os.urandom(12)
        aad = b"context-info"
        ct, tag = chacha20_poly1305_encrypt(key, nonce, b"data", aad=aad)
        with pytest.raises(ValueError, match="tag mismatch"):
            chacha20_poly1305_decrypt(key, nonce, ct, tag, aad=b"wrong-context")

    def test_wrong_key_raises(self):
        key = os.urandom(32)
        nonce = os.urandom(12)
        ct, tag = chacha20_poly1305_encrypt(key, nonce, b"data")
        wrong_key = os.urandom(32)
        with pytest.raises(ValueError, match="tag mismatch"):
            chacha20_poly1305_decrypt(wrong_key, nonce, ct, tag)

    def test_invalid_key_length(self):
        with pytest.raises(ValueError, match="32 bytes"):
            chacha20_poly1305_encrypt(b"short", os.urandom(12), b"data")

    def test_invalid_nonce_length(self):
        with pytest.raises(ValueError, match="12 bytes"):
            chacha20_poly1305_encrypt(os.urandom(32), b"short", b"data")

    def test_large_plaintext(self):
        key = os.urandom(32)
        nonce = os.urandom(12)
        plaintext = os.urandom(4096)
        ct, tag = chacha20_poly1305_encrypt(key, nonce, plaintext)
        recovered = chacha20_poly1305_decrypt(key, nonce, ct, tag)
        assert recovered == plaintext
