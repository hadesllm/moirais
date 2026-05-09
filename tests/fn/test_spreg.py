"""Tests for moirais.fn.spreg — spatial regimes."""
import numpy as np
from moirais.fn.spreg import spatial_regime


class TestSpatialRegime:
    def test_basic(self):
        rng = np.random.default_rng(42)
        n = 40
        X = rng.standard_normal((n, 2))
        regions = np.array(["A"] * 20 + ["B"] * 20)
        y = X @ [1, 0.5] + rng.standard_normal(n)
        res = spatial_regime(y, X, regions)
        assert res.extra["n_regimes"] == 2

    def test_pvalue_valid(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((30, 1))
        y = rng.standard_normal(30)
        regions = np.repeat([1, 2, 3], 10)
        res = spatial_regime(y, X, regions)
        assert 0 <= res.extra["p_value"] <= 1
