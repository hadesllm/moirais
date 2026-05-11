"""Test skewness_coeff (sskew)."""
import numpy as np
from morie.fn.sskew import skewness_coeff, sskew
from morie.fn._containers import DescriptiveResult


class TestSkewness:
    def test_symmetric(self):
        x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        result = skewness_coeff(x)
        assert isinstance(result, DescriptiveResult)
        assert abs(result.value) < 1e-10

    def test_positive_skew(self):
        x = np.array([1.0, 1.0, 1.0, 1.0, 10.0])
        assert skewness_coeff(x).value > 0

    def test_alias(self):
        assert sskew is skewness_coeff
