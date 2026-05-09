"""Test svm_rbf (svmrb)."""
import numpy as np
from moirais.fn.svmrb import svm_rbf, svmrb
from moirais.fn._containers import DescriptiveResult


class TestSvmrb:
    def test_basic(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((60, 3))
        y = (X[:, 0] ** 2 + X[:, 1] ** 2 > 1).astype(int)
        result = svm_rbf(X[:50], y[:50], X[50:])
        assert isinstance(result, DescriptiveResult)
        assert result.name == "svm_rbf"

    def test_support_vectors(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((60, 3))
        y = (X[:, 0] > 0).astype(int)
        result = svm_rbf(X[:50], y[:50], X[50:])
        assert result.extra["n_support_vectors"] > 0

    def test_alias(self):
        assert svmrb is svm_rbf
