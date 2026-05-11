"""Tests for kendall_tau_b."""
import numpy as np, pytest
from morie.fn.kendt import kendall_tau_b

class TestKendall:
    def test_perfect(self):
        x = np.arange(10, dtype=float)
        r = kendall_tau_b(x, x)
        assert r.estimate == pytest.approx(1.0)

    def test_ci(self):
        rng = np.random.default_rng(0)
        x = rng.normal(0, 1, 50)
        y = x + rng.normal(0, 0.5, 50)
        r = kendall_tau_b(x, y)
        assert r.ci_lower < r.estimate < r.ci_upper
