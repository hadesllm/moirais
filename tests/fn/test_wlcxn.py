"""Tests for moirais.fn.wlcxn — Wilcoxon survival test."""

import numpy as np
import pytest

from moirais.fn.wlcxn import wilcoxon_survival


class TestWilcoxonSurvival:
    def test_identical(self):
        rng = np.random.default_rng(42)
        t = rng.exponential(5, 50)
        e = np.ones(50, dtype=int)
        res = wilcoxon_survival(t, e, t, e)
        assert res.p_value > 0.05

    def test_different(self):
        rng = np.random.default_rng(42)
        t1 = rng.exponential(2, 100)
        t2 = rng.exponential(10, 100)
        e = np.ones(100, dtype=int)
        res = wilcoxon_survival(t1, e, t2, e)
        assert res.p_value < 0.05

    def test_method(self):
        t1 = np.array([1, 2, 3, 4, 5.0])
        e1 = np.ones(5, dtype=int)
        res = wilcoxon_survival(t1, e1, t1 + 1, e1)
        assert res.method == "Gehan-Breslow"
