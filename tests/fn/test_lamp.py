"""Test lamport_sign."""
from moirais.fn._containers import CryptoResult
from moirais.fn.lamp import lamp, lamport_sign


class TestLamportSign:
    def test_basic(self):
        result = lamport_sign(b"hello")
        assert isinstance(result, CryptoResult)
        assert result.success is True
        assert result.algorithm == "Lamport-OTS"
        assert result.operation == "sign"

    def test_extra_has_signature_and_pk(self):
        result = lamport_sign(b"hello")
        assert "signature" in result.extra
        assert "pk" in result.extra
        assert "sk" in result.extra

    def test_alias(self):
        assert lamp is lamport_sign
