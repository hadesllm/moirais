"""Tests for moirais.fn.spcae -- Sparse PCA."""

import numpy as np
from moirais.fn.spcae import sparse_pca, spcae
from moirais.fn._containers import PcaRes


class TestSparsePca:
    def test_alias(self):
        assert spcae is sparse_pca

    def test_returns_pca_res(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((50, 5))
        res = sparse_pca(X, n_components=2, alpha=0.5)
        assert isinstance(res, PcaRes)

    def test_sparsity(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((50, 10))
        res = sparse_pca(X, n_components=2, alpha=2.0)
        n_zeros = np.sum(np.abs(res.components) < 1e-10)
        assert n_zeros > 0

    def test_shapes(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((30, 6))
        res = sparse_pca(X, n_components=3)
        assert res.components.shape == (6, 3)
        assert res.scores.shape == (30, 3)
