"""Test detrend_signal (dtrnd)."""
import numpy as np
from morie.fn.dtrnd import detrend_signal, dtrnd
from morie.fn._containers import SignalResult


class TestDetrendSignal:
    def test_basic(self):
        x = np.arange(100, dtype=float) + np.random.default_rng(42).standard_normal(100)
        result = detrend_signal(x)
        assert isinstance(result, SignalResult)
        assert result.name == "detrend_signal"

    def test_removes_linear_trend(self):
        x = np.arange(100, dtype=float) * 2.0 + 5.0
        result = detrend_signal(x, order=1)
        assert np.std(result.filtered) < 1e-8

    def test_alias(self):
        assert dtrnd is detrend_signal
