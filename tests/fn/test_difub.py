"""Tests for difub -- Uniform DIF."""
import numpy as np
from moirais.fn.difub import dif_uniform
from moirais.fn._containers import DIFResult


class TestDifUniform:
    def test_basic(self):
        rng = np.random.default_rng(42)
        n = 200
        X = rng.binomial(1, 0.5, (n, 5))
        total = X.sum(axis=1).astype(float)
        group = np.array([0] * 100 + [1] * 100)
        result = dif_uniform(X, total, group)
        assert isinstance(result, DIFResult)
        assert result.method == "Uniform"

    def test_group_coef_col(self):
        rng = np.random.default_rng(42)
        X = rng.binomial(1, 0.5, (100, 3))
        total = X.sum(axis=1).astype(float)
        group = np.array([0] * 50 + [1] * 50)
        result = dif_uniform(X, total, group)
        assert "group_coef" in result.items.columns
