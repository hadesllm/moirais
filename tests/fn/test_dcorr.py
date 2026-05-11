"""Tests for distance_correlation."""
import numpy as np, pytest
from morie.fn.dcorr import distance_correlation

class TestDcorr:
    def test_correlated(self):
        rng = np.random.default_rng(0)
        x = rng.normal(0, 1, 30)
        y = x ** 2 + rng.normal(0, 0.1, 30)
        r = distance_correlation(x, y)
        assert r.estimate > 0.5

    def test_independent(self):
        rng = np.random.default_rng(1)
        x = rng.normal(0, 1, 50)
        y = rng.normal(0, 1, 50)
        r = distance_correlation(x, y)
        assert r.estimate < 0.5
