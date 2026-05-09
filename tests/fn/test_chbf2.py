"""Test chebyshev2_filter (chbf2)."""
import numpy as np
from moirais.fn.chbf2 import chebyshev2_filter, chbf2
from moirais.fn._containers import SignalResult


class TestChebyshev2Filter:
    def test_basic(self):
        x = np.random.default_rng(42).standard_normal(256)
        result = chebyshev2_filter(x, cutoff=50.0, fs=500.0)
        assert isinstance(result, SignalResult)
        assert result.name == "chebyshev2_filter"

    def test_output_shape(self):
        x = np.random.default_rng(42).standard_normal(256)
        result = chebyshev2_filter(x, cutoff=50.0, fs=500.0)
        assert result.n_samples == 256
        assert len(result.filtered) == 256

    def test_alias(self):
        assert chbf2 is chebyshev2_filter
