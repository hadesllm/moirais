"""Test chebyshev1_filter (chbf1)."""
import numpy as np
from moirais.fn.chbf1 import chebyshev1_filter, chbf1
from moirais.fn._containers import SignalResult


class TestChebyshev1Filter:
    def test_basic(self):
        x = np.random.default_rng(42).standard_normal(256)
        result = chebyshev1_filter(x, cutoff=50.0, fs=500.0)
        assert isinstance(result, SignalResult)
        assert result.name == "chebyshev1_filter"

    def test_output_shape(self):
        x = np.random.default_rng(42).standard_normal(256)
        result = chebyshev1_filter(x, cutoff=50.0, fs=500.0)
        assert result.n_samples == 256
        assert len(result.filtered) == 256

    def test_smoothing(self):
        rng = np.random.default_rng(42)
        x = np.sin(np.linspace(0, 10, 500)) + rng.standard_normal(500)
        result = chebyshev1_filter(x, cutoff=20.0, fs=500.0)
        assert np.std(result.filtered) < np.std(x)

    def test_alias(self):
        assert chbf1 is chebyshev1_filter
