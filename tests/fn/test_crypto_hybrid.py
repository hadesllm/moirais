"""Tests for moirais.crypto.hybrid — Hybrid KEM-DEM encryption."""

import os

import pytest

from moirais.crypto.hybrid import hybrid_decrypt, hybrid_encrypt, keygen


class TestHybridEncryption:

    @pytest.fixture(scope="class")
    def keypair(self):
        return keygen()

    def test_roundtrip_small(self, keypair):
        pk, sk = keypair
        plaintext = b"Do. Or do not. There is no try."
        ct = hybrid_encrypt(plaintext, pk)
        recovered = hybrid_decrypt(ct, sk)
        assert recovered == plaintext

    def test_roundtrip_empty(self, keypair):
        pk, sk = keypair
        ct = hybrid_encrypt(b"", pk)
        recovered = hybrid_decrypt(ct, sk)
        assert recovered == b""

    def test_roundtrip_10kb(self, keypair):
        pk, sk = keypair
        plaintext = os.urandom(10 * 1024)
        ct = hybrid_encrypt(plaintext, pk)
        recovered = hybrid_decrypt(ct, sk)
        assert recovered == plaintext

    def test_wrong_key_fails(self, keypair):
        pk, _sk = keypair
        _, wrong_sk = keygen()
        ct = hybrid_encrypt(b"secret", pk)
        with pytest.raises((ValueError, Exception)):
            hybrid_decrypt(ct, wrong_sk)

    def test_tampered_ciphertext_fails(self, keypair):
        pk, sk = keypair
        ct = hybrid_encrypt(b"important data", pk)
        tampered = bytearray(ct)
        tampered[-17] ^= 0xFF
        with pytest.raises((ValueError, Exception)):
            hybrid_decrypt(bytes(tampered), sk)

    def test_truncated_ciphertext_fails(self, keypair):
        _pk, sk = keypair
        with pytest.raises(ValueError):
            hybrid_decrypt(b"\x00\x00", sk)

    def test_ciphertext_larger_than_plaintext(self, keypair):
        pk, _sk = keypair
        plaintext = b"x"
        ct = hybrid_encrypt(plaintext, pk)
        assert len(ct) > len(plaintext)
