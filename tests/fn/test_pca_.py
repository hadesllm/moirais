"""Tests for morie.fn.pca_ — PCA."""
import numpy as np
import pytest
from morie.fn.pca_ import pca_simple


class TestPCA:
    def test_basic(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((50, 5))
        res = pca_simple(X, n_components=2)
        assert res.extra["scores"].shape == (50, 2)
        assert res.extra["loadings"].shape == (5, 2)

    def test_variance_ratio_sums_leq_one(self):
        X = np.random.default_rng(42).standard_normal((30, 4))
        res = pca_simple(X, n_components=2)
        assert res.value <= 1.0 + 1e-10
