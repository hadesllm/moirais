"""Tests for moirais.fn.rskci — risk concordance."""

import pytest
import numpy as np
from moirais.fn.rskci import risk_concordance
from moirais.fn._containers import ESRes


class TestRiskConcordance:

    def test_returns_esres(self):
        scores = np.array([5, 3, 1, 4, 2])
        times = np.array([1, 2, 5, 3, 4], dtype=float)
        events = np.array([1, 1, 0, 1, 0])
        result = risk_concordance(scores, times, events)
        assert isinstance(result, ESRes)

    def test_c_bounded(self):
        rng = np.random.default_rng(42)
        n = 30
        scores = rng.standard_normal(n)
        times = rng.exponential(3, n)
        events = rng.integers(0, 2, n)
        result = risk_concordance(scores, times, events)
        assert 0 <= result.estimate <= 1
