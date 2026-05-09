"""Tests for canonical_correlation."""
import numpy as np, pytest
from moirais.fn.cca import canonical_correlation

class TestCCA:
    def test_correlated(self):
        rng = np.random.default_rng(0)
        z = rng.normal(0, 1, (50, 2))
        X = z + rng.normal(0, 0.3, (50, 2))
        Y = z + rng.normal(0, 0.3, (50, 2))
        r = canonical_correlation(X, Y)
        assert r.value > 0.5

    def test_independent(self):
        rng = np.random.default_rng(1)
        X = rng.normal(0, 1, (50, 2))
        Y = rng.normal(0, 1, (50, 2))
        r = canonical_correlation(X, Y)
        assert r.name == "cca"
