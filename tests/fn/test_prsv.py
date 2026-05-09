"""Test parseval_verify (prsv)."""
import numpy as np
from moirais.fn.prsv import parseval_verify, prsv
from moirais.fn._containers import DescriptiveResult


class TestPrsv:
    def test_basic(self):
        x = np.array([1.0, 2.0, 3.0, 4.0])
        result = parseval_verify(x)
        assert isinstance(result, DescriptiveResult)
        assert result.name == "parseval_verify"

    def test_ratio_unity(self):
        x = np.random.default_rng(42).standard_normal(64)
        result = parseval_verify(x)
        assert np.isclose(result.value, 1.0, atol=1e-10)

    def test_alias(self):
        assert prsv is parseval_verify
