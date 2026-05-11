"""Tests for morie.fn.epdsg -- epidemic curve fitting."""

import numpy as np
import pytest
from morie.fn.epdsg import epidemic_curve_fit


class TestEpiCurveFit:
    def test_lognormal_peak(self):
        from scipy.stats import lognorm
        t = np.arange(60, dtype=float)
        inc = 500 * lognorm.pdf(t, s=0.5, scale=np.exp(2.5))
        res = epidemic_curve_fit(inc, distribution="lognormal")
        assert 8 < res["peak_time"] < 20
        assert res["rmse"] < 5.0

    def test_gamma_fit(self):
        from scipy.stats import gamma
        t = np.arange(50, dtype=float)
        inc = 300 * gamma.pdf(t, a=5, scale=3)
        res = epidemic_curve_fit(inc, distribution="gamma")
        assert res["peak_time"] > 5
        assert res["distribution"] == "gamma"

    def test_total_cases(self):
        inc = np.array([1, 3, 8, 15, 20, 18, 10, 5, 2, 1], dtype=float)
        res = epidemic_curve_fit(inc)
        assert res["total_cases"] == pytest.approx(83.0)

    def test_short_array_raises(self):
        with pytest.raises(ValueError):
            epidemic_curve_fit(np.array([1.0, 2.0]))

    def test_invalid_dist_raises(self):
        with pytest.raises(ValueError):
            epidemic_curve_fit(np.ones(10), distribution="weibull")
