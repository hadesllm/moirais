"""Test mldsa_verify."""
from moirais.fn._containers import CryptoResult
from moirais.fn.mldsa import mldsa_keygen
from moirais.fn.mldss import mldsa_sign
from moirais.fn.mldsv import mldsa_verify, mldsv


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
