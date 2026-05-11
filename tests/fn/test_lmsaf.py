"""Test lms_adaptive_filter (lmsaf)."""
import numpy as np
import pytest

from morie.fn.lmsaf import lms_adaptive_filter, lmsaf
from morie.fn._containers import SignalResult


class TestLmsAdaptiveFilter:
    def test_basic(self):
        x = np.random.default_rng(42).standard_normal(256)
        d = x + 0.1 * np.random.default_rng(43).standard_normal(256)
        result = lms_adaptive_filter(x, d)
        assert isinstance(result, SignalResult)
        assert result.name == "lms_adaptive_filter"

    def test_output_shape(self):
        x = np.random.default_rng(42).standard_normal(256)
        d = x + 0.1 * np.random.default_rng(43).standard_normal(256)
        result = lms_adaptive_filter(x, d)
        assert result.n_samples == 256

    def test_extra_has_output_error(self):
        x = np.random.default_rng(42).standard_normal(256)
        d = x + 0.1 * np.random.default_rng(43).standard_normal(256)
        result = lms_adaptive_filter(x, d)
        assert "output" in result.extra
        assert "error" in result.extra

    def test_order_param(self):
        x = np.random.default_rng(42).standard_normal(256)
        d = x + 0.1 * np.random.default_rng(43).standard_normal(256)
        result = lms_adaptive_filter(x, d, order=8, mu=0.005)
        assert isinstance(result, SignalResult)

    def test_alias(self):
        assert lmsaf is lms_adaptive_filter
