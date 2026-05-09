"""Tests for moirais.fn.pcaev -- PCA via eigenvalue decomposition."""

import numpy as np
from moirais.fn.pcaev import pca_eigen, pcaev
from moirais.fn._containers import PcaRes


class TestPcaEigen:
    def test_alias(self):
        assert pcaev is pca_eigen

    def test_returns_pca_res(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((50, 5))
        res = pca_eigen(X, n_components=2)
        assert isinstance(res, PcaRes)

    def test_shapes(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((50, 5))
        res = pca_eigen(X, n_components=3)
        assert res.components.shape == (5, 3)
        assert res.scores.shape == (50, 3)
        assert len(res.explained_variance) == 3

    def test_variance_ratio_sums_le_one(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((100, 4))
        res = pca_eigen(X)
        assert np.sum(res.explained_variance_ratio) <= 1.0 + 1e-10

    def test_eigenvalues_positive(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((60, 3))
        res = pca_eigen(X)
        assert np.all(res.explained_variance >= -1e-10)
