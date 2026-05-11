"""Test wots_verify."""
from morie.fn._containers import CryptoResult
from morie.fn.wots import wots_sign
from morie.fn.wotsv import wots_verify, wotsv


class TestWotsVerify:
    def test_roundtrip(self):
        signed = wots_sign(b"test")
        result = wots_verify(
            b"test",
            signed.extra["signature"],
            signed.extra["pk"],
        )
        assert isinstance(result, CryptoResult)
        assert result.success is True

    def test_alias(self):
        assert wotsv is wots_verify
