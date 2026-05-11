"""Test rls_filter (rlsfl)."""
import numpy as np
from morie.fn.rlsfl import rls_filter, rlsfl
from morie.fn._containers import SignalResult


class TestRlsFilter:
    def test_basic(self):
        rng = np.random.default_rng(42)
        x = rng.standard_normal(500)
        d = np.convolve(x, [1.0, 0.5], mode="full")[:500]
        result = rls_filter(x, d, lam=0.99, order=8)
        assert isinstance(result, SignalResult)
        assert result.name == "rls_filter"

    def test_has_weights(self):
        rng = np.random.default_rng(42)
        x = rng.standard_normal(500)
        d = np.convolve(x, [1.0, 0.3], mode="full")[:500]
        result = rls_filter(x, d, lam=0.99, order=4)
        assert "weights" in result.extra
        assert len(result.extra["weights"]) == 4

    def test_alias(self):
        assert rlsfl is rls_filter
