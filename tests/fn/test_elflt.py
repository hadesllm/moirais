"""Test elliptic_filter (elflt)."""
import numpy as np
from moirais.fn.elflt import elliptic_filter, elflt
from moirais.fn._containers import SignalResult


class TestEllipticFilter:
    def test_basic(self):
        x = np.random.default_rng(42).standard_normal(256)
        result = elliptic_filter(x, cutoff=50.0, fs=500.0)
        assert isinstance(result, SignalResult)
        assert result.name == "elliptic_filter"

    def test_output_shape(self):
        x = np.random.default_rng(42).standard_normal(256)
        result = elliptic_filter(x, cutoff=50.0, fs=500.0)
        assert result.n_samples == 256
        assert len(result.filtered) == 256

    def test_alias(self):
        assert elflt is elliptic_filter
