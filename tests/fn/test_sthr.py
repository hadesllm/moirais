"""Test soft_threshold (sthr)."""
import numpy as np
from moirais.fn.sthr import soft_threshold, sthr
from moirais.fn._containers import DescriptiveResult


class TestSthr:
    def test_basic(self):
        x = np.array([0.5, -0.3, 0.05, 1.0, -0.8])
        result = soft_threshold(x, lambda_=0.2)
        assert isinstance(result, DescriptiveResult)
        assert result.name == "soft_threshold"

    def test_known_values(self):
        x = np.array([1.0, -0.5, 0.1, -0.1, 0.3])
        r = soft_threshold(x, lambda_=0.2)
        expected = np.array([0.8, -0.3, 0.0, 0.0, 0.1])
        np.testing.assert_allclose(r.extra["thresholded"], expected, atol=1e-10)

    def test_zeros_below_threshold(self):
        x = np.array([0.05, -0.05, 0.01])
        r = soft_threshold(x, lambda_=0.1)
        np.testing.assert_allclose(r.extra["thresholded"], np.zeros(3), atol=1e-10)
        assert r.value == 0

    def test_n_zeroed(self):
        x = np.array([1.0, 0.05, -2.0, 0.01])
        r = soft_threshold(x, lambda_=0.1)
        assert r.extra["n_zeroed"] == 2

    def test_alias(self):
        assert sthr is soft_threshold
