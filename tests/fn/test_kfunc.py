"""Tests for moirais.fn.kfunc — Ripley's K."""
import numpy as np
from moirais.fn.kfunc import ripley_k


class TestRipleyK:
    def test_basic(self):
        pts = np.random.default_rng(42).uniform(0, 10, (30, 2))
        res = ripley_k(pts, n_distances=10)
        assert len(res.extra["K"]) == 10

    def test_k_increasing(self):
        pts = np.random.default_rng(42).uniform(0, 10, (30, 2))
        res = ripley_k(pts, n_distances=10)
        K = res.extra["K"]
        assert all(K[i] <= K[i+1] + 1e-10 for i in range(len(K)-1))
