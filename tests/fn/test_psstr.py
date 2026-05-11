"""Tests for propensity_stratify."""
import numpy as np, pytest
from morie.fn.psstr import propensity_stratify

class TestPSStratify:
    def test_basic(self):
        rng = np.random.default_rng(0)
        n = 100
        ps = rng.uniform(0.1, 0.9, n)
        t = (rng.random(n) < ps).astype(float)
        y = t * 2 + rng.normal(0, 1, n)
        r = propensity_stratify(ps, t, y)
        assert r.extra["n_strata"] == 5

    def test_ate_direction(self):
        rng = np.random.default_rng(1)
        n = 200
        ps = rng.uniform(0.2, 0.8, n)
        t = (rng.random(n) < ps).astype(float)
        y = t * 5 + rng.normal(0, 1, n)
        r = propensity_stratify(ps, t, y)
        assert r.value > 0
