"""Tests for latent_class."""
import numpy as np, pytest
from moirais.fn.lca import latent_class

class TestLCA:
    def test_two_classes(self):
        rng = np.random.default_rng(0)
        c1 = (rng.random((30, 5)) < 0.8).astype(float)
        c2 = (rng.random((30, 5)) < 0.2).astype(float)
        X = np.vstack([c1, c2])
        r = latent_class(X, n_classes=2, seed=0)
        assert r.name == "lca"
        assert sum(r.extra["class_sizes"]) == 60

    def test_bic_finite(self):
        rng = np.random.default_rng(1)
        X = (rng.random((40, 4)) < 0.5).astype(float)
        r = latent_class(X, n_classes=2, seed=1)
        assert np.isfinite(r.value)
