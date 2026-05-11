"""Tests for runs_test."""
import numpy as np, pytest
from morie.fn.rnsum import runs_test

class TestRuns:
    def test_random(self):
        rng = np.random.default_rng(0)
        x = rng.normal(0, 1, 50)
        r = runs_test(x)
        assert r.test_name == "Runs"
        assert r.p_value > 0.01

    def test_structured(self):
        x = np.concatenate([np.ones(15), np.zeros(15)])
        r = runs_test(x)
        assert r.p_value < 0.05
