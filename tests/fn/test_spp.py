"""Test peak_to_peak (spp)."""
import numpy as np
from morie.fn.spp import peak_to_peak, spp
from morie.fn._containers import DescriptiveResult


class TestPeakToPeak:
    def test_basic(self):
        x = np.array([-3.0, 0.0, 5.0])
        result = peak_to_peak(x)
        assert isinstance(result, DescriptiveResult)
        assert abs(result.value - 8.0) < 1e-10

    def test_constant(self):
        x = np.array([2.0, 2.0, 2.0])
        assert peak_to_peak(x).value == 0.0

    def test_alias(self):
        assert spp is peak_to_peak
