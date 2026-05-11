"""Test mldsa_sign."""
from morie.fn._containers import CryptoResult
from morie.fn.mldsa import mldsa_keygen
from morie.fn.mldss import mldsa_sign, mldss


class TestMldsaSign:
    def test_basic(self):
        keys = mldsa_keygen()
        result = mldsa_sign(b"test", keys.extra["sk"])
        assert isinstance(result, CryptoResult)
        assert result.success is True
        assert result.algorithm == "ML-DSA"
        assert result.operation == "sign"

    def test_extra_has_signature(self):
        keys = mldsa_keygen()
        result = mldsa_sign(b"test", keys.extra["sk"])
        assert "signature" in result.extra
        assert result.extra["sig_len"] > 0

    def test_alias(self):
        assert mldss is mldsa_sign
