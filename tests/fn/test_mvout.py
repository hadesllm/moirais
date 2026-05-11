"""Tests for multivariate_outlier."""
import numpy as np, pytest
from morie.fn.mvout import multivariate_outlier


class TestMultivariateOutlier:
    def test_no_outliers(self):
        rng = np.random.default_rng(42)
        X = rng.normal(0, 1, (50, 3))
        r = multivariate_outlier(X)
        assert r.measure == "multivariate_outlier"
        assert r.estimate < 10

    def test_with_outlier(self):
        rng = np.random.default_rng(42)
        X = rng.normal(0, 1, (50, 2))
        X[0] = [100, 100]
        r = multivariate_outlier(X)
        assert 0 in r.extra["outlier_indices"]

    def test_too_few(self):
        with pytest.raises(ValueError):
            multivariate_outlier(np.array([[1, 2]]))
