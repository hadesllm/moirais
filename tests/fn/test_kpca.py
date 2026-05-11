"""Tests for morie.fn.kpca -- Kernel PCA."""

import numpy as np
from morie.fn.kpca import kernel_pca, kpca
from morie.fn._containers import PcaRes


class TestKernelPca:
    def test_alias(self):
        assert kpca is kernel_pca

    def test_returns_pca_res(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((40, 3))
        res = kernel_pca(X, n_components=2)
        assert isinstance(res, PcaRes)

    def test_rbf_kernel(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((40, 3))
        res = kernel_pca(X, kernel="rbf", n_components=2)
        assert res.scores.shape == (40, 2)

    def test_linear_kernel(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((30, 4))
        res = kernel_pca(X, kernel="linear", n_components=2)
        assert res.scores.shape == (30, 2)

    def test_poly_kernel(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((30, 3))
        res = kernel_pca(X, kernel="poly", degree=2, n_components=2)
        assert res.scores.shape == (30, 2)
