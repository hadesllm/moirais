"""Tests for min_covariance_det."""
import numpy as np, pytest
from moirais.fn.mcd import min_covariance_det

class TestMCD:
    def test_clean_data(self):
        rng = np.random.default_rng(0)
        X = rng.normal(0, 1, (50, 2))
        r = min_covariance_det(X, seed=0)
        assert r.name == "mcd"
        assert len(r.extra["location"]) == 2

    def test_with_outliers(self):
        rng = np.random.default_rng(1)
        X = rng.normal(0, 1, (50, 2))
        X[:3] = 100
        r = min_covariance_det(X, seed=1)
        assert abs(r.extra["location"][0]) < 5
