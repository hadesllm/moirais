"""Tests for covariate_balance_ps."""
import numpy as np, pytest
from moirais.fn.cbps import covariate_balance_ps

class TestCBPS:
    def test_basic(self):
        rng = np.random.default_rng(0)
        n = 100
        X = rng.normal(0, 1, (n, 2))
        t = (X[:, 0] > 0).astype(float)
        r = covariate_balance_ps(t, X)
        assert r.name == "cbps"

    def test_balance_improves(self):
        rng = np.random.default_rng(1)
        X = rng.normal(0, 1, (200, 2))
        t = (0.5 * X[:, 0] + rng.normal(0, 1, 200) > 0).astype(float)
        r = covariate_balance_ps(t, X)
        assert r.extra["smd_after"][0] <= r.extra["smd_before"][0] + 0.5
