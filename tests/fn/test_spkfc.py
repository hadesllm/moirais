"""Test peak_factor (spkfc)."""
import numpy as np
from moirais.fn.spkfc import peak_factor, spkfc
from moirais.fn._containers import DescriptiveResult


class TestPeakFactor:
    def test_dc(self):
        x = np.array([5.0, 5.0, 5.0])
        result = peak_factor(x)
        assert isinstance(result, DescriptiveResult)
        assert abs(result.value - 1.0) < 1e-10

    def test_alias(self):
        assert spkfc is peak_factor
