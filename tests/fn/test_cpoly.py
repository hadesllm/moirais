"""Tests for cpoly — ChaCha20-Poly1305 authenticated encryption."""
import os
import pytest
from morie.fn.cpoly import chacha20_encrypt, chacha20_decrypt
from morie.fn._containers import CryptoResult


def test_cpoly_roundtrip():
    key = os.urandom(32)
    nonce = os.urandom(12)
    plaintext = b"Hello, post-quantum world!"
    enc = chacha20_encrypt(key, nonce, plaintext)
    assert isinstance(enc, CryptoResult)
    assert enc.success is True
    ct = enc.extra["ciphertext"]
    tag = enc.extra["tag"]
    assert ct != plaintext
    dec = chacha20_decrypt(key, nonce, ct, tag)
    assert dec.extra["plaintext"] == plaintext


def test_cpoly_aad():
    key = os.urandom(32)
    nonce = os.urandom(12)
    plaintext = b"secret data"
    aad = b"authenticated header"
    enc = chacha20_encrypt(key, nonce, plaintext, aad=aad)
    ct, tag = enc.extra["ciphertext"], enc.extra["tag"]
    dec = chacha20_decrypt(key, nonce, ct, tag, aad=aad)
    assert dec.extra["plaintext"] == plaintext


def test_cpoly_tampered_ciphertext():
    key = os.urandom(32)
    nonce = os.urandom(12)
    enc = chacha20_encrypt(key, nonce, b"test data")
    ct = bytearray(enc.extra["ciphertext"])
    ct[0] ^= 0xFF
    with pytest.raises(ValueError, match="Authentication failed"):
        chacha20_decrypt(key, nonce, bytes(ct), enc.extra["tag"])


def test_cpoly_tampered_aad():
    key = os.urandom(32)
    nonce = os.urandom(12)
    enc = chacha20_encrypt(key, nonce, b"test", aad=b"real")
    with pytest.raises(ValueError, match="Authentication failed"):
        chacha20_decrypt(key, nonce, enc.extra["ciphertext"], enc.extra["tag"], aad=b"fake")


def test_cpoly_wrong_key_length():
    with pytest.raises(ValueError, match="Key must be 32"):
        chacha20_encrypt(b"short", os.urandom(12), b"data")
