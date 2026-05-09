"""Tests for finite_mixture."""
import numpy as np, pytest
from moirais.fn.fmm import finite_mixture

class TestFMM:
    def test_two_clusters(self):
        rng = np.random.default_rng(0)
        X = np.vstack([rng.normal(0, 1, (30, 2)), rng.normal(5, 1, (30, 2))])
        r = finite_mixture(X, n_components=2, seed=0)
        assert r.name == "fmm"
        assert sum(r.extra["class_sizes"]) == 60

    def test_bic(self):
        rng = np.random.default_rng(1)
        X = rng.normal(0, 1, (40, 2))
        r = finite_mixture(X, n_components=2, seed=1)
        assert np.isfinite(r.value)
