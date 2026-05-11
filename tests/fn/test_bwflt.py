"""Test butterworth_filter (bwflt)."""
import numpy as np
from morie.fn.bwflt import butterworth_filter, bwflt
from morie.fn._containers import SignalResult


class TestButterworthFilter:
    def test_basic(self):
        rng = np.random.default_rng(42)
        x = rng.standard_normal(256)
        result = butterworth_filter(x, cutoff=50.0, fs=500.0)
        assert isinstance(result, SignalResult)
        assert result.name == "butterworth_filter"

    def test_output_shape(self):
        x = np.random.default_rng(42).standard_normal(256)
        result = butterworth_filter(x, cutoff=50.0, fs=500.0)
        assert result.n_samples == 256
        assert result.filtered is not None
        assert len(result.filtered) == 256

    def test_lowpass_smoothing(self):
        rng = np.random.default_rng(42)
        t = np.linspace(0, 1, 1000)
        x = np.sin(2 * np.pi * 5 * t) + 0.5 * rng.standard_normal(1000)
        result = butterworth_filter(x, cutoff=20.0, fs=1000.0, order=4)
        assert np.std(result.filtered) < np.std(x)

    def test_alias(self):
        assert bwflt is butterworth_filter
