"""Test svm_poly (svmpl)."""
import numpy as np
from morie.fn.svmpl import svm_poly, svmpl
from morie.fn._containers import DescriptiveResult


class TestSvmpl:
    def test_basic(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((60, 3))
        y = (X[:, 0] > 0).astype(int)
        result = svm_poly(X[:50], y[:50], X[50:])
        assert isinstance(result, DescriptiveResult)
        assert result.name == "svm_poly"

    def test_predictions_shape(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((60, 3))
        y = (X[:, 0] > 0).astype(int)
        result = svm_poly(X[:50], y[:50], X[50:])
        assert len(result.extra["predictions"]) == 10

    def test_alias(self):
        assert svmpl is svm_poly
