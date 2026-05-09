"""Tests for mahalanobis_distance."""
import numpy as np, pytest
from moirais.fn.mrdst import mahalanobis_distance

class TestMahalanobis:
    def test_identity_cov(self):
        x = np.array([[3, 0]])
        r = mahalanobis_distance(x, mean=np.array([0, 0]), cov=np.eye(2))
        assert r.extra["distances"][0] == pytest.approx(3.0)

    def test_sample(self):
        rng = np.random.default_rng(0)
        X = rng.normal(0, 1, (50, 3))
        r = mahalanobis_distance(X)
        assert r.value > 0
