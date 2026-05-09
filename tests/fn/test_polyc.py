"""Tests for polychoric_corr."""
import numpy as np, pytest
from moirais.fn.polyc import polychoric_corr

class TestPolychoric:
    def test_positive(self):
        rng = np.random.default_rng(0)
        z = rng.normal(0, 1, 100)
        x = np.digitize(z, [-1, 0, 1])
        y = np.digitize(z + rng.normal(0, 0.5, 100), [-1, 0, 1])
        r = polychoric_corr(x, y)
        assert r.estimate > 0.3

    def test_independent(self):
        rng = np.random.default_rng(1)
        x = rng.integers(0, 4, 100)
        y = rng.integers(0, 4, 100)
        r = polychoric_corr(x, y)
        assert abs(r.estimate) < 0.4
