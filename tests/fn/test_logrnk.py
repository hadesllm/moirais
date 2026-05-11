"""Tests for morie.fn.logrnk — Log-rank test."""

import numpy as np
import pytest

from morie.fn.logrnk import logrank_test


class TestLogRank:
    def test_identical_groups(self):
        rng = np.random.default_rng(42)
        t = rng.exponential(5, 50)
        e = np.ones(50, dtype=int)
        res = logrank_test(t, e, t, e)
        assert res.p_value > 0.05

    def test_different_groups(self):
        rng = np.random.default_rng(42)
        t1 = rng.exponential(2, 100)
        t2 = rng.exponential(10, 100)
        e = np.ones(100, dtype=int)
        res = logrank_test(t1, e, t2, e)
        assert res.p_value < 0.05

    def test_returns_test_result(self):
        t1 = np.array([1, 2, 3, 4, 5.0])
        e1 = np.array([1, 1, 1, 0, 1])
        t2 = np.array([2, 4, 6, 8, 10.0])
        e2 = np.array([1, 1, 0, 1, 1])
        res = logrank_test(t1, e1, t2, e2)
        assert res.test_name == "log_rank"
