"""Test lamport_verify."""
from morie.fn._containers import CryptoResult
from morie.fn.lamp import lamport_sign
from morie.fn.lampv import lamport_verify, lampv


class TestLamportVerify:
    def test_valid_signature(self):
        signed = lamport_sign(b"hello")
        result = lamport_verify(
            b"hello",
            signed.extra["signature"],
            signed.extra["pk"],
        )
        assert isinstance(result, CryptoResult)
        assert result.success is True

    def test_wrong_message(self):
        signed = lamport_sign(b"hello")
        result = lamport_verify(
            b"wrong",
            signed.extra["signature"],
            signed.extra["pk"],
        )
        assert result.success is False

    def test_alias(self):
        assert lampv is lamport_verify
