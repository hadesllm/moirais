"""Test bandstop_filter (bsflt)."""
import numpy as np
from morie.fn.bsflt import bandstop_filter, bsflt
from morie.fn._containers import SignalResult


class TestBandstopFilter:
    def test_basic(self):
        x = np.random.default_rng(42).standard_normal(512)
        result = bandstop_filter(x, low=20.0, high=80.0, fs=500.0)
        assert isinstance(result, SignalResult)
        assert result.name == "bandstop_filter"

    def test_output_shape(self):
        x = np.random.default_rng(42).standard_normal(512)
        result = bandstop_filter(x, low=20.0, high=80.0, fs=500.0)
        assert result.n_samples == 512
        assert len(result.filtered) == 512

    def test_alias(self):
        assert bsflt is bandstop_filter
