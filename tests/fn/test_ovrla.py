"""Tests for overlap_weight."""
import numpy as np, pytest
from moirais.fn.ovrla import overlap_weight

class TestOverlap:
    def test_ate_direction(self):
        rng = np.random.default_rng(0)
        n = 200
        ps = rng.uniform(0.2, 0.8, n)
        t = (rng.random(n) < ps).astype(float)
        y = t * 3 + rng.normal(0, 1, n)
        r = overlap_weight(ps, t, y)
        assert r.value > 0

    def test_balanced(self):
        ps = np.full(100, 0.5)
        t = np.array([0]*50 + [1]*50, dtype=float)
        y = t * 2
        r = overlap_weight(ps, t, y)
        assert r.value == pytest.approx(2.0, abs=0.1)
