"""Tests for moirais.fn.rcomp -- Rademacher complexity computation."""

import numpy as np
import pytest
from moirais.fn.rcomp import rademacher_complexity


class TestRademacherComplexity:
    def test_basic_output(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((50, 3))
        r = rademacher_complexity(X)
        assert r["rademacher_complexity"] > 0
        assert r["n"] == 50
        assert r["d"] == 3

    def test_1d_input(self):
        r = rademacher_complexity(np.array([1.0, 2.0, 3.0]))
        assert r["d"] == 1
        assert r["n"] == 3

    def test_higher_d_larger_complexity(self):
        rng = np.random.default_rng(42)
        r1 = rademacher_complexity(rng.standard_normal((100, 1)))["rademacher_complexity"]
        r10 = rademacher_complexity(rng.standard_normal((100, 10)))["rademacher_complexity"]
        assert r10 > r1

    def test_deterministic_with_seed(self):
        X = np.ones((20, 2))
        r1 = rademacher_complexity(X, seed=99)
        r2 = rademacher_complexity(X, seed=99)
        assert r1["rademacher_complexity"] == r2["rademacher_complexity"]

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            rademacher_complexity(np.array([]))
