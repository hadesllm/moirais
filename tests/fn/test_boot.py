"""Tests for morie.fn.boot -- bootstrap confidence intervals."""

import numpy as np
import pandas as pd
import pytest
from morie.fn.boot import bootstrap_ci


class TestBootstrapCI:
    def test_mean_ci(self):
        """Bootstrap CI for mean of normal data should contain true mean."""
        data = pd.DataFrame({"x": np.random.default_rng(42).normal(5.0, 1.0, 200)})
        lo, hi = bootstrap_ci(lambda df: df["x"].mean(), data, n_iterations=500)
        assert lo < 5.0 < hi

    def test_returns_tuple(self):
        """Should return a (lower, upper) tuple."""
        data = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})
        result = bootstrap_ci(lambda df: df["x"].mean(), data, n_iterations=100)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert result[0] < result[1]

    def test_reproducible(self):
        """Same seed should give same result."""
        data = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]})
        r1 = bootstrap_ci(lambda df: df["x"].mean(), data, seed=123)
        r2 = bootstrap_ci(lambda df: df["x"].mean(), data, seed=123)
        assert r1[0] == pytest.approx(r2[0])
        assert r1[1] == pytest.approx(r2[1])
