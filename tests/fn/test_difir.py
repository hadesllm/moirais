"""Tests for difir -- IRT-based DIF."""
import numpy as np
from morie.fn.difir import dif_irt_based
from morie.fn._containers import DIFResult


class TestDifIrt:
    def test_basic(self):
        rng = np.random.default_rng(42)
        n = 200
        X = rng.binomial(1, 0.6, (n, 5))
        group = np.array([0] * 100 + [1] * 100)
        result = dif_irt_based(X, group)
        assert isinstance(result, DIFResult)
        assert result.method == "IRT-LR"

    def test_has_items_df(self):
        rng = np.random.default_rng(42)
        X = rng.binomial(1, 0.5, (100, 3))
        group = np.array([0] * 50 + [1] * 50)
        result = dif_irt_based(X, group)
        assert "b_ref" in result.items.columns
