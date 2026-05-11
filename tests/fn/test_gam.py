"""Tests for morie.fn.gam -- GAM (B-spline + OLS)."""

import numpy as np
from morie.fn.gam import fit_gam, gam
from morie.fn._containers import DescriptiveResult


class TestGam:
    def test_alias(self):
        assert gam is fit_gam

    def test_sin_curve(self):
        rng = np.random.default_rng(42)
        x = np.linspace(0, 2 * np.pi, 100)
        y = np.sin(x) + rng.normal(0, 0.1, 100)
        result = fit_gam(x, y, n_splines=10)
        assert isinstance(result, DescriptiveResult)
        assert result.extra["r_squared"] > 0.8

    def test_returns_predicted(self):
        x = np.linspace(0, 1, 50)
        y = x ** 2
        result = fit_gam(x, y)
        assert len(result.extra["predicted"]) == 50
