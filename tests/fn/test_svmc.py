"""Tests for moirais.fn.svmc — SVM classifier."""
import numpy as np
import pytest
from moirais.fn.svmc import svm_classify


class TestSVMClassify:
    def test_xor_rbf(self):
        rng = np.random.default_rng(42)
        n = 80
        X = rng.standard_normal((n, 2))
        y = np.where(X[:, 0] * X[:, 1] > 0, 1.0, -1.0)
        res = svm_classify(X, y, kernel="rbf", C=10.0)
        assert res.extra["accuracy"] > 0.5

    def test_support_vectors(self):
        rng = np.random.default_rng(42)
        n = 60
        X = rng.standard_normal((n, 2))
        y = np.where(X[:, 0] > 0, 1.0, -1.0)
        res = svm_classify(X, y, kernel="linear")
        assert res.extra["n_support_vectors"] > 0

    def test_linear_separable(self):
        rng = np.random.default_rng(42)
        X = np.vstack([rng.normal(-2, 0.5, (30, 2)),
                        rng.normal(2, 0.5, (30, 2))])
        y = np.array([-1.0] * 30 + [1.0] * 30)
        res = svm_classify(X, y, kernel="linear")
        assert res.extra["accuracy"] > 0.8
