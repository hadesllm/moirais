"""Tests for morie.fn.mrvsn -- Granger causality."""

import numpy as np
from morie.fn.mrvsn import granger_causality, mrvsn
from morie.fn._containers import TestResult


class TestMrvsn:
    def test_alias(self):
        assert mrvsn is granger_causality

    def test_causal_relationship(self):
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, 200)
        y = np.zeros(200)
        for t in range(2, 200):
            y[t] = 0.8 * x[t - 1] + rng.normal(0, 0.1)
        result = granger_causality(x, y, max_lag=2)
        assert isinstance(result, TestResult)
        assert result.p_value < 0.05

    def test_no_causality(self):
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, 200)
        y = rng.normal(0, 1, 200)
        result = granger_causality(x, y, max_lag=2)
        assert result.p_value > 0.01
