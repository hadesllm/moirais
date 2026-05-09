"""Tests for moirais.fn.pwexp — Piecewise exponential model."""

import numpy as np
import pytest

from moirais.fn.pwexp import piecewise_exponential


class TestPiecewiseExponential:
    def test_basic(self):
        rng = np.random.default_rng(42)
        times = rng.exponential(5, 100)
        events = rng.choice([0, 1], 100, p=[0.3, 0.7])
        res = piecewise_exponential(times, events)
        assert len(res.extra["hazard_rates"]) >= 2

    def test_custom_breaks(self):
        rng = np.random.default_rng(42)
        times = rng.exponential(5, 100)
        events = np.ones(100, dtype=int)
        res = piecewise_exponential(times, events, breaks=[2, 5, 10])
        assert len(res.extra["intervals"]) == 4

    def test_length_mismatch(self):
        with pytest.raises(ValueError):
            piecewise_exponential([1, 2], [1])
