"""Test wots_sign."""
from moirais.fn._containers import CryptoResult
from moirais.fn.wots import wots, wots_sign


class TestWotsSign:
    def test_basic(self):
        result = wots_sign(b"test")
        assert isinstance(result, CryptoResult)
        assert result.success is True
        assert result.algorithm == "WOTS+"
        assert result.operation == "sign"

    def test_extra_has_signature(self):
        result = wots_sign(b"test")
        assert "signature" in result.extra
        assert "pk" in result.extra
        assert "w" in result.extra

    def test_alias(self):
        assert wots is wots_sign
