"""Tests for morie.fn.gmm — Gaussian mixture model."""
import numpy as np
from morie.fn.gmm import gaussian_mixture


class TestGMM:
    def test_basic(self):
        rng = np.random.default_rng(42)
        X = np.vstack([rng.normal(0, 1, (40, 2)), rng.normal(5, 1, (40, 2))])
        res = gaussian_mixture(X, n_components=2)
        assert len(np.unique(res.extra["labels"])) == 2

    def test_weights_sum_one(self):
        X = np.random.default_rng(42).standard_normal((50, 2))
        res = gaussian_mixture(X, n_components=3)
        np.testing.assert_allclose(res.extra["weights"].sum(), 1.0, atol=1e-6)
