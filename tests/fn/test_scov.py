"""Test sample_covariance (scov)."""
import numpy as np
from moirais.fn.scov import sample_covariance, scov
from moirais.fn._containers import DescriptiveResult


class TestSampleCovariance:
    def test_perfect_positive(self):
        x = np.array([1.0, 2.0, 3.0])
        result = sample_covariance(x, x)
        assert isinstance(result, DescriptiveResult)
        assert abs(result.value - np.var(x, ddof=1)) < 1e-10

    def test_uncorrelated(self):
        x = np.array([1.0, -1.0, 1.0, -1.0])
        y = np.array([1.0, 1.0, -1.0, -1.0])
        assert abs(sample_covariance(x, y).value) < 1e-10

    def test_alias(self):
        assert scov is sample_covariance
