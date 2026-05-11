"""Test raw_moment (smom1)."""
import numpy as np
from morie.fn.smom1 import raw_moment, smom1
from morie.fn._containers import DescriptiveResult


class TestRawMoment:
    def test_first_moment_is_mean(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = raw_moment(x, k=1)
        assert isinstance(result, DescriptiveResult)
        assert abs(result.value - 3.0) < 1e-10

    def test_second_moment(self):
        x = np.array([1.0, 2.0, 3.0])
        assert abs(raw_moment(x, k=2).value - np.mean(x ** 2)) < 1e-10

    def test_alias(self):
        assert smom1 is raw_moment
