"""Tests for morie.fn.nnidx — nearest neighbor index."""
import numpy as np
from morie.fn.nnidx import nearest_neighbor_index


class TestNNIndex:
    def test_random_points(self):
        pts = np.random.default_rng(42).uniform(0, 10, (50, 2))
        res = nearest_neighbor_index(pts)
        assert res.statistic > 0
        assert 0 <= res.p_value <= 1

    def test_clustered(self):
        pts = np.random.default_rng(42).normal(5, 0.1, (30, 2))
        res = nearest_neighbor_index(pts)
        assert res.statistic < 1.5
