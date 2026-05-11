"""Tests for morie.fn.theil — Theil-Sen estimator."""
import numpy as np
import pytest
from morie.fn.theil import theil_sen


class TestTheilSen:
    def test_linear(self):
        x = np.arange(50, dtype=float)
        y = 2.0 * x + 1.0
        res = theil_sen(x, y)
        assert res.estimate == pytest.approx(2.0, abs=0.01)

    def test_ci_contains_slope(self):
        rng = np.random.default_rng(42)
        x = np.arange(30, dtype=float)
        y = 1.5 * x + rng.standard_normal(30) * 2
        res = theil_sen(x, y)
        assert res.ci_lower <= res.estimate <= res.ci_upper
