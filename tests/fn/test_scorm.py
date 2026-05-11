"""Test covariance_matrix (scorm)."""
import numpy as np
from morie.fn.scorm import covariance_matrix, scorm
from morie.fn._containers import DescriptiveResult


class TestCovarianceMatrix:
    def test_shape(self):
        X = np.random.default_rng(42).normal(size=(50, 3))
        result = covariance_matrix(X)
        assert isinstance(result, DescriptiveResult)
        assert result.value.shape == (3, 3)

    def test_symmetric(self):
        X = np.random.default_rng(42).normal(size=(20, 4))
        C = covariance_matrix(X).value
        assert np.allclose(C, C.T)

    def test_alias(self):
        assert scorm is covariance_matrix
