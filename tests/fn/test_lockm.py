"""Tests for morie.fn.lockm -- SHA-256 hash."""

from morie.fn.lockm import sha256_hash, lockm
from morie.fn._containers import DescriptiveResult


class TestLockm:
    def test_alias(self):
        assert lockm is sha256_hash

    def test_known_hash(self):
        result = sha256_hash("hello")
        assert isinstance(result, DescriptiveResult)
        assert result.value == "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"

    def test_iterated(self):
        r1 = sha256_hash("test", iterations=1)
        r2 = sha256_hash("test", iterations=2)
        assert r1.value != r2.value
        assert len(r2.value) == 64
