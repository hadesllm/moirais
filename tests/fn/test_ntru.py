"""Test ntru_keygen."""
from morie.fn._containers import CryptoResult
from morie.fn.ntru import ntru, ntru_keygen


class TestNtruKeygen:
    def test_basic(self):
        result = ntru_keygen(n=11, q=32)
        assert isinstance(result, CryptoResult)
        assert result.success is True
        assert result.algorithm == "NTRU"
        assert result.operation == "keygen"

    def test_extra_has_keys(self):
        result = ntru_keygen(n=11, q=32)
        assert "pk" in result.extra
        assert "sk" in result.extra

    def test_alias(self):
        assert ntru is ntru_keygen
