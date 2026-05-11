"""Tests for difnb -- Non-uniform DIF."""
import numpy as np
from morie.fn.difnb import dif_nonuniform
from morie.fn._containers import DIFResult


class TestDifNonuniform:
    def test_basic(self):
        rng = np.random.default_rng(42)
        n = 200
        X = rng.binomial(1, 0.5, (n, 5))
        ability = X.sum(axis=1).astype(float)
        group = np.array([0] * 100 + [1] * 100)
        result = dif_nonuniform(X, ability, group)
        assert isinstance(result, DIFResult)
        assert result.method == "NonUniform"

    def test_interaction_col(self):
        rng = np.random.default_rng(42)
        X = rng.binomial(1, 0.5, (100, 3))
        ability = X.sum(axis=1).astype(float)
        group = np.array([0] * 50 + [1] * 50)
        result = dif_nonuniform(X, ability, group)
        assert "interaction_coef" in result.items.columns
