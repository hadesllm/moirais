"""Test kurtosis_coeff (skurt)."""
import numpy as np
from morie.fn.skurt import kurtosis_coeff, skurt
from morie.fn._containers import DescriptiveResult


class TestKurtosis:
    def test_normal_like(self):
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, 10000)
        result = kurtosis_coeff(x)
        assert isinstance(result, DescriptiveResult)
        assert abs(result.value) < 0.3

    def test_alias(self):
        assert skurt is kurtosis_coeff
