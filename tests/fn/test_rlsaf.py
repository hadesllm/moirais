"""Test rls_adaptive_filter (rlsaf)."""
import numpy as np
import pytest

from morie.fn.rlsaf import rls_adaptive_filter, rlsaf
from morie.fn._containers import SignalResult


class TestRlsAdaptiveFilter:
    def test_basic(self):
        x = np.random.default_rng(42).standard_normal(256)
        d = x + 0.1 * np.random.default_rng(43).standard_normal(256)
        result = rls_adaptive_filter(x, d)
        assert isinstance(result, SignalResult)
        assert result.name == "rls_adaptive_filter"

    def test_output_shape(self):
        x = np.random.default_rng(42).standard_normal(256)
        d = x + 0.1 * np.random.default_rng(43).standard_normal(256)
        result = rls_adaptive_filter(x, d)
        assert result.n_samples == 256

    def test_extra_has_output_error(self):
        x = np.random.default_rng(42).standard_normal(256)
        d = x + 0.1 * np.random.default_rng(43).standard_normal(256)
        result = rls_adaptive_filter(x, d)
        assert "output" in result.extra
        assert "error" in result.extra

    def test_params(self):
        x = np.random.default_rng(42).standard_normal(256)
        d = x + 0.1 * np.random.default_rng(43).standard_normal(256)
        result = rls_adaptive_filter(x, d, order=8, lam=0.98, delta=50.0)
        assert isinstance(result, SignalResult)

    def test_alias(self):
        assert rlsaf is rls_adaptive_filter
