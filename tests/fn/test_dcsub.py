"""Test dc_removal (dcsub)."""
import numpy as np
from morie.fn.dcsub import dc_removal, dcsub
from morie.fn._containers import SignalResult


class TestDcRemoval:
    def test_basic(self):
        x = np.array([5.0, 6.0, 7.0, 8.0, 9.0])
        result = dc_removal(x)
        assert isinstance(result, SignalResult)
        assert result.name == "dc_removal"

    def test_zero_mean(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = dc_removal(x)
        assert abs(np.mean(result.filtered)) < 1e-10

    def test_alias(self):
        assert dcsub is dc_removal
