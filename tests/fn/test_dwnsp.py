"""Test downsample (dwnsp)."""
import numpy as np
from morie.fn.dwnsp import downsample, dwnsp
from morie.fn._containers import SignalResult


class TestDownsample:
    def test_basic(self):
        x = np.arange(100, dtype=float)
        result = downsample(x, factor=2)
        assert isinstance(result, SignalResult)
        assert result.n_samples == 50

    def test_values(self):
        x = np.arange(10, dtype=float)
        result = downsample(x, factor=3)
        np.testing.assert_array_equal(result.filtered, [0, 3, 6, 9])

    def test_alias(self):
        assert dwnsp is downsample
