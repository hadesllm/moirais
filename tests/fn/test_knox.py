"""Tests for moirais.fn.knox — Knox test."""
import numpy as np
from moirais.fn.knox import knox_test


class TestKnox:
    def test_basic(self):
        rng = np.random.default_rng(42)
        locs = rng.uniform(0, 10, (20, 2))
        times = rng.uniform(0, 100, 20)
        res = knox_test(locs, times, s_threshold=3, t_threshold=20, n_permutations=99)
        assert 0 <= res.p_value <= 1

    def test_stat_nonneg(self):
        rng = np.random.default_rng(42)
        locs = rng.uniform(0, 5, (15, 2))
        times = rng.uniform(0, 50, 15)
        res = knox_test(locs, times, s_threshold=2, t_threshold=10, n_permutations=49)
        assert res.statistic >= 0
