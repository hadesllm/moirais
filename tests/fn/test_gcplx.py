"""Tests for moirais.fn.gcplx -- Gaussian complexity computation."""

import numpy as np
import pytest
from moirais.fn.gcplx import gaussian_complexity


class TestGaussianComplexity:
    def test_basic_output(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((50, 3))
        r = gaussian_complexity(X)
        assert r["gaussian_complexity"] > 0
        assert r["n"] == 50
        assert r["d"] == 3

    def test_rademacher_upper_bound(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((100, 5))
        r = gaussian_complexity(X)
        assert r["rademacher_upper"] == pytest.approx(
            r["gaussian_complexity"] * np.sqrt(np.pi / 2)
        )

    def test_1d_input(self):
        r = gaussian_complexity(np.array([1.0, 2.0, 3.0, 4.0]))
        assert r["d"] == 1

    def test_deterministic(self):
        X = np.ones((20, 2))
        r1 = gaussian_complexity(X, seed=7)
        r2 = gaussian_complexity(X, seed=7)
        assert r1["gaussian_complexity"] == r2["gaussian_complexity"]

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            gaussian_complexity(np.array([]))
