"""Test mldsa_keygen."""
from moirais.fn._containers import CryptoResult
from moirais.fn.mldsa import mldsa, mldsa_keygen


class TestMldsaKeygen:
    def test_basic(self):
        result = mldsa_keygen()
        assert isinstance(result, CryptoResult)
        assert result.success is True
        assert result.algorithm == "ML-DSA"
        assert result.operation == "keygen"

    def test_extra_has_keys(self):
        result = mldsa_keygen()
        assert "pk" in result.extra
        assert "sk" in result.extra
        assert result.extra["pk_len"] > 0
        assert result.extra["sk_len"] > 0

    def test_alias(self):
        assert mldsa is mldsa_keygen
