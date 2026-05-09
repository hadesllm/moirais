"""Test goppa_code."""
from moirais.fn._containers import CryptoResult
from moirais.fn.goppa import goppa, goppa_code


class TestGoppaCode:
    def test_basic(self):
        result = goppa_code(m=4, t=2)
        assert isinstance(result, CryptoResult)
        assert result.success is True
        assert result.algorithm == "Goppa"
        assert result.operation == "generate"

    def test_extra_has_H(self):
        result = goppa_code(m=4, t=2)
        assert "H" in result.extra

    def test_alias(self):
        assert goppa is goppa_code
