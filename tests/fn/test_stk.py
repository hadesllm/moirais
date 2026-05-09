"""Tests for moirais.fn.stk — space-time K."""
import numpy as np
from moirais.fn.stk import space_time_k


class TestSpaceTimeK:
    def test_basic(self):
        rng = np.random.default_rng(42)
        pts = rng.uniform(0, 10, (30, 2))
        times = rng.uniform(0, 100, 30)
        res = space_time_k(pts, times, n_s=5, n_t=5)
        assert res.extra["K_st"].shape == (5, 5)

    def test_nonneg(self):
        rng = np.random.default_rng(42)
        pts = rng.uniform(0, 5, (20, 2))
        times = rng.uniform(0, 50, 20)
        res = space_time_k(pts, times, n_s=3, n_t=3)
        assert np.all(res.extra["K_st"] >= 0)
