"""Tests for moirais.fn.pcsvd -- PCA via SVD."""

import numpy as np
from moirais.fn.pcsvd import pca_svd, pcsvd
from moirais.fn._containers import PcaRes


class TestPcaSvd:
    def test_alias(self):
        assert pcsvd is pca_svd

    def test_returns_pca_res(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((50, 5))
        res = pca_svd(X, n_components=2)
        assert isinstance(res, PcaRes)

    def test_shapes(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((50, 5))
        res = pca_svd(X, n_components=3)
        assert res.components.shape == (5, 3)
        assert res.scores.shape == (50, 3)

    def test_agrees_with_eigen(self):
        from moirais.fn.pcaev import pca_eigen
        rng = np.random.default_rng(42)
        X = rng.standard_normal((80, 4))
        r1 = pca_eigen(X, n_components=2)
        r2 = pca_svd(X, n_components=2)
        np.testing.assert_allclose(
            r1.explained_variance, r2.explained_variance, atol=1e-8
        )

    def test_variance_ratio_valid(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((100, 4))
        res = pca_svd(X)
        assert np.all(res.explained_variance_ratio >= 0)
        assert np.sum(res.explained_variance_ratio) <= 1.0 + 1e-10
