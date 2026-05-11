"""Test central_moment (smom2)."""
import numpy as np
from morie.fn.smom2 import central_moment, smom2
from morie.fn._containers import DescriptiveResult


class TestCentralMoment:
    def test_second_is_variance(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = central_moment(x, k=2)
        assert isinstance(result, DescriptiveResult)
        assert abs(result.value - np.var(x, ddof=0)) < 1e-10

    def test_first_is_zero(self):
        x = np.array([1.0, 2.0, 3.0])
        assert abs(central_moment(x, k=1).value) < 1e-10

    def test_alias(self):
        assert smom2 is central_moment
