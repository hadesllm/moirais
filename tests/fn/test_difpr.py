"""Tests for difpr -- DIF purification."""
import numpy as np
from morie.fn.difpr import dif_purification
from morie.fn._containers import DIFResult


class TestDifPurification:
    def test_basic(self):
        rng = np.random.default_rng(42)
        X = rng.binomial(1, 0.5, (200, 5))
        group = np.array([0] * 100 + [1] * 100)
        result = dif_purification(X, group)
        assert isinstance(result, DIFResult)
        assert result.method == "Purification"

    def test_anchor_column(self):
        rng = np.random.default_rng(42)
        X = rng.binomial(1, 0.5, (100, 4))
        group = np.array([0] * 50 + [1] * 50)
        result = dif_purification(X, group)
        assert "anchor" in result.items.columns
