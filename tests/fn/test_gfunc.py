"""Tests for moirais.fn.gfunc — pair correlation."""
import numpy as np
from moirais.fn.gfunc import pair_correlation


class TestPairCorrelation:
    def test_basic(self):
        pts = np.random.default_rng(42).uniform(0, 10, (30, 2))
        res = pair_correlation(pts, n_distances=10)
        assert len(res.extra["g"]) == 10

    def test_non_negative(self):
        pts = np.random.default_rng(42).uniform(0, 5, (20, 2))
        res = pair_correlation(pts)
        assert all(g >= 0 for g in res.extra["g"])
