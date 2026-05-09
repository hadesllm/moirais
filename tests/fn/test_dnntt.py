"""Tests for dunnett_test."""
import numpy as np, pytest
from moirais.fn.dnntt import dunnett_test

class TestDunnett:
    def test_basic(self):
        ctrl = np.array([1,2,3,4,5], dtype=float)
        t1 = np.array([5,6,7,8,9], dtype=float)
        t2 = np.array([2,3,4,5,6], dtype=float)
        r = dunnett_test(ctrl, t1, t2)
        assert r.name == "dunnett"
        assert len(r.extra["comparisons"]) == 2

    def test_no_diff(self):
        rng = np.random.default_rng(0)
        ctrl = rng.normal(0, 1, 20)
        t1 = rng.normal(0, 1, 20)
        r = dunnett_test(ctrl, t1)
        assert r.extra["comparisons"][0]["p_adj"] > 0.01
