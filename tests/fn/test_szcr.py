"""Test zero_crossing_rate (szcr)."""
import numpy as np
from morie.fn.szcr import zero_crossing_rate, szcr
from morie.fn._containers import DescriptiveResult


class TestZCR:
    def test_alternating(self):
        x = np.array([1.0, -1.0, 1.0, -1.0])
        result = zero_crossing_rate(x)
        assert isinstance(result, DescriptiveResult)
        assert result.value == 1.0

    def test_no_crossings(self):
        x = np.array([1.0, 2.0, 3.0])
        assert zero_crossing_rate(x).value == 0.0

    def test_alias(self):
        assert szcr is zero_crossing_rate
