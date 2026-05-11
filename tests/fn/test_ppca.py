"""Tests for probabilistic_pca."""
import numpy as np, pytest
from morie.fn.ppca import probabilistic_pca

class TestPPCA:
    def test_basic(self):
        rng = np.random.default_rng(0)
        X = rng.normal(0, 1, (50, 5))
        r = probabilistic_pca(X, n_components=2, seed=0)
        assert r.name == "ppca"
        assert r.extra["q"] == 2

    def test_sigma_positive(self):
        rng = np.random.default_rng(1)
        X = rng.normal(0, 1, (30, 3))
        r = probabilistic_pca(X, n_components=1, seed=1)
        assert r.value > 0
