"""Tests for morie.fn.scan — spatial scan statistic."""
import numpy as np
from morie.fn.scan import spatial_scan


class TestSpatialScan:
    def test_basic(self):
        rng = np.random.default_rng(42)
        coords = rng.uniform(0, 10, (15, 2))
        pops = rng.integers(100, 1000, 15).astype(float)
        counts = rng.poisson(5, 15).astype(float)
        counts[0] = 50
        res = spatial_scan(counts, pops, coords, n_simulations=19)
        assert res.extra["llr"] >= 0

    def test_pvalue_bounded(self):
        rng = np.random.default_rng(42)
        coords = rng.uniform(0, 5, (10, 2))
        pops = np.ones(10) * 100
        counts = np.ones(10) * 5
        res = spatial_scan(counts, pops, coords, n_simulations=9)
        assert 0 <= res.extra["p_value"] <= 1
