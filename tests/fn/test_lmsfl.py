"""Test lms_filter (lmsfl)."""
import numpy as np
from morie.fn.lmsfl import lms_filter, lmsfl
from morie.fn._containers import SignalResult


class TestLmsFilter:
    def test_basic(self):
        rng = np.random.default_rng(42)
        x = rng.standard_normal(500)
        d = np.convolve(x, [1.0, 0.5, 0.25], mode="full")[:500]
        result = lms_filter(x, d, mu=0.01, order=8)
        assert isinstance(result, SignalResult)
        assert result.name == "lms_filter"

    def test_has_weights(self):
        rng = np.random.default_rng(42)
        x = rng.standard_normal(500)
        d = np.convolve(x, [1.0, 0.5, 0.25], mode="full")[:500]
        result = lms_filter(x, d, mu=0.01, order=8)
        assert "weights" in result.extra
        assert len(result.extra["weights"]) == 8

    def test_alias(self):
        assert lmsfl is lms_filter
