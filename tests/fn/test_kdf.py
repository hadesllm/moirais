"""Tests for moirais.crypto._kdf — HKDF-SHA256."""

import pytest

from moirais.crypto._kdf import hkdf_sha256


class TestHKDF:

    def test_rfc5869_vector1(self):
        """RFC 5869 Test Case 1 for HKDF-SHA256."""
        ikm = bytes.fromhex("0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b")
        salt = bytes.fromhex("000102030405060708090a0b0c")
        info = bytes.fromhex("f0f1f2f3f4f5f6f7f8f9")
        expected = bytes.fromhex(
            "3cb25f25faacd57a90434f64d0362f2a"
            "2d2d0a90cf1a5a4c5db02d56ecc4c5bf"
            "34007208d5b887185865"
        )
        result = hkdf_sha256(ikm, length=42, salt=salt, info=info)
        assert result == expected

    def test_deterministic(self):
        ikm = b"input keying material"
        a = hkdf_sha256(ikm, length=32, info=b"ctx")
        b = hkdf_sha256(ikm, length=32, info=b"ctx")
        assert a == b

    def test_different_lengths(self):
        ikm = b"same input"
        out16 = hkdf_sha256(ikm, length=16)
        out32 = hkdf_sha256(ikm, length=32)
        assert len(out16) == 16
        assert len(out32) == 32
        assert out16 != out32[:16] or len(out16) != len(out32)

    def test_different_info_different_output(self):
        ikm = b"same input"
        a = hkdf_sha256(ikm, length=32, info=b"context-a")
        b = hkdf_sha256(ikm, length=32, info=b"context-b")
        assert a != b

    def test_empty_salt(self):
        ikm = b"input"
        result = hkdf_sha256(ikm, length=32, salt=b"")
        assert len(result) == 32

    def test_length_exceeds_max_raises(self):
        with pytest.raises(ValueError, match="exceeds maximum"):
            hkdf_sha256(b"x", length=255 * 32 + 1)
