"""Test mldsa_verify."""
from morie.fn._containers import CryptoResult
from morie.fn.mldsa import mldsa_keygen
from morie.fn.mldss import mldsa_sign
from morie.fn.mldsv import mldsa_verify, mldsv


class TestMldsaVerify:
    def test_roundtrip(self):
        keys = mldsa_keygen()
        signed = mldsa_sign(b"test", keys.extra["sk"])
        result = mldsa_verify(
            b"test",
            signed.extra["signature"],
            keys.extra["pk"],
        )
        assert isinstance(result, CryptoResult)
        assert result.success is True
        assert result.extra["valid"] is True

    def test_alias(self):
        assert mldsv is mldsa_verify
