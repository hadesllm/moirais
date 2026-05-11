"""Test hard_threshold (hthr)."""
import numpy as np
from morie.fn.hthr import hard_threshold, hthr
from morie.fn._containers import DescriptiveResult


class TestHthr:
    def test_basic(self):
        x = np.array([0.5, -0.3, 0.05, 1.0, -0.8])
        result = hard_threshold(x, lambda_=0.2)
        assert isinstance(result, DescriptiveResult)
        assert result.name == "hard_threshold"

    def test_known_values(self):
        x = np.array([1.0, -0.5, 0.1, -0.1, 0.3])
        r = hard_threshold(x, lambda_=0.2)
        expected = np.array([1.0, -0.5, 0.0, 0.0, 0.3])
        np.testing.assert_allclose(r.extra["thresholded"], expected, atol=1e-10)

    def test_zeros_below_threshold(self):
        x = np.array([0.05, -0.05, 0.01])
        r = hard_threshold(x, lambda_=0.1)
        np.testing.assert_allclose(r.extra["thresholded"], np.zeros(3), atol=1e-10)
        assert r.value == 0

    def test_preserves_large(self):
        x = np.array([5.0, -3.0])
        r = hard_threshold(x, lambda_=0.1)
        np.testing.assert_allclose(r.extra["thresholded"], x, atol=1e-10)
        assert r.value == 2

    def test_alias(self):
        assert hthr is hard_threshold
