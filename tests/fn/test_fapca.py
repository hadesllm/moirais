"""Tests for factor_pca_compare."""
import numpy as np, pytest
from morie.fn.fapca import factor_pca_compare

class TestFaPca:
    def test_basic(self):
        rng = np.random.default_rng(0)
        X = rng.normal(0, 1, (50, 5))
        r = factor_pca_compare(X, n_components=2)
        assert len(r.extra["pca_eigenvalues"]) == 2
        assert len(r.extra["communalities"]) == 5

    def test_var_explained(self):
        rng = np.random.default_rng(1)
        X = rng.normal(0, 1, (30, 3))
        r = factor_pca_compare(X, n_components=2)
        assert r.value > 0
