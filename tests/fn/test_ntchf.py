"""Test notch_filter (ntchf)."""
import numpy as np
from moirais.fn.ntchf import notch_filter, ntchf
from moirais.fn._containers import SignalResult


class TestNotchFilter:
    def test_basic(self):
        t = np.linspace(0, 1, 1000)
        x = np.sin(2 * np.pi * 50 * t) + np.sin(2 * np.pi * 120 * t)
        result = notch_filter(x, freq=120.0, Q=30.0, fs=1000.0)
        assert isinstance(result, SignalResult)
        assert result.name == "notch_filter"

    def test_output_shape(self):
        x = np.random.default_rng(42).standard_normal(500)
        result = notch_filter(x, freq=0.1, Q=10.0, fs=1.0)
        assert result.n_samples == 500
        assert len(result.filtered) == 500

    def test_alias(self):
        assert ntchf is notch_filter
