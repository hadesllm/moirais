"""Test sample_variance (svar)."""
import numpy as np
from moirais.fn.svar import sample_variance, svar
from moirais.fn._containers import DescriptiveResult


class TestSampleVariance:
    def test_basic(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = sample_variance(x)
        assert isinstance(result, DescriptiveResult)
        assert abs(result.value - 2.5) < 1e-10

    def test_ddof_zero(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = sample_variance(x, ddof=0)
        assert abs(result.value - 2.0) < 1e-10

    def test_alias(self):
        assert svar is sample_variance
