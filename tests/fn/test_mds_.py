"""Tests for metric_mds."""
import numpy as np, pytest
from moirais.fn.mds_ import metric_mds

class TestMDS:
    def test_basic(self):
        D = np.array([[0,1,2],[1,0,1.5],[2,1.5,0]], dtype=float)
        r = metric_mds(D, n_dims=2)
        assert r.name == "mds"
        assert r.value >= 0

    def test_euclidean(self):
        pts = np.array([[0,0],[1,0],[0,1],[1,1]], dtype=float)
        D = np.sqrt(np.sum((pts[:, None] - pts[None, :]) ** 2, axis=2))
        r = metric_mds(D, n_dims=2)
        assert r.value < 0.1
