"""Tests for morie.fn.gxenv -- Gene-by-environment interaction."""

import numpy as np
import pytest
from morie.fn.gxenv import gxenv


class TestGxenv:
    def test_no_interaction(self):
        rng = np.random.default_rng(42)
        n = 100
        g = rng.choice([0, 1, 2], size=n).astype(float)
        env = rng.standard_normal(n)
        y = 1.0 + 0.5 * g + 0.3 * env + rng.standard_normal(n) * 0.5
        res = gxenv(y, g, env)
        assert res.p_value > 0.05

    def test_strong_interaction(self):
        rng = np.random.default_rng(42)
        n = 200
        g = rng.choice([0, 1, 2], size=n).astype(float)
        env = rng.standard_normal(n)
        y = 1.0 + 3.0 * g * env + rng.standard_normal(n) * 0.3
        res = gxenv(y, g, env)
        assert res.p_value < 0.05

    def test_f_stat_nonnegative(self):
        rng = np.random.default_rng(42)
        y = rng.standard_normal(50)
        g = rng.choice([0, 1, 2], size=50).astype(float)
        env = rng.standard_normal(50)
        res = gxenv(y, g, env)
        assert res.statistic >= 0

    def test_dimension_mismatch(self):
        with pytest.raises(ValueError):
            gxenv(np.ones(10), np.ones(10), np.ones(5))
