"""Tests for negbin_regression."""
import numpy as np, pytest
from morie.fn.nbreg import negbin_regression

class TestNegBin:
    def test_basic(self):
        rng = np.random.default_rng(0)
        X = rng.normal(0, 1, (80, 1))
        y = rng.negative_binomial(5, 0.5, 80)
        r = negbin_regression(y, X)
        assert r.name == "negbin"
        assert r.extra["alpha"] > 0

    def test_aic(self):
        rng = np.random.default_rng(1)
        y = rng.negative_binomial(3, 0.3, 60)
        X = rng.normal(0, 1, (60, 1))
        r = negbin_regression(y, X)
        assert np.isfinite(r.extra["aic"])
