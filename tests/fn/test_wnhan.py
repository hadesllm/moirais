"""Test hanning_window (wnhan)."""
import numpy as np
from moirais.fn.wnhan import hanning_window, wnhan
from moirais.fn._containers import DescriptiveResult


class TestWnhan:
    def test_basic(self):
        result = hanning_window(16)
        assert isinstance(result, DescriptiveResult)
        assert result.name == "hanning_window"

    def test_zero_endpoints(self):
        result = hanning_window(32)
        w = result.extra["window"]
        assert np.isclose(w[0], 0.0, atol=1e-10)

    def test_alias(self):
        assert wnhan is hanning_window
