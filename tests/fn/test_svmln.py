"""Test svm_linear (svmln)."""
import numpy as np
from moirais.fn.svmln import svm_linear, svmln
from moirais.fn._containers import DescriptiveResult


class TestSvmln:
    def test_basic(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((60, 3))
        y = (X[:, 0] > 0).astype(int)
        result = svm_linear(X[:50], y[:50], X[50:])
        assert isinstance(result, DescriptiveResult)
        assert result.name == "svm_linear"
        assert "predictions" in result.extra

    def test_predictions_shape(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((60, 3))
        y = (X[:, 0] > 0).astype(int)
        result = svm_linear(X[:50], y[:50], X[50:])
        assert len(result.extra["predictions"]) == 10

    def test_alias(self):
        assert svmln is svm_linear
