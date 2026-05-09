"""Tests for moirais.fn.epcrv -- Epidemic curve analysis."""

import pytest
from moirais.fn.epcrv import epidemic_curve_analysis


class TestEpidemicCurveAnalysis:
    def test_basic(self):
        inc = [1, 2, 4, 8, 16, 32, 20, 10, 5, 3, 1]
        res = epidemic_curve_analysis(inc)
        assert res.measure == "epidemic_curve_analysis"
        assert res.extra["peak_day"] == 5

    def test_growth_rate(self):
        inc = [1, 2, 4, 8, 16, 32, 20, 10, 5, 3, 1]
        res = epidemic_curve_analysis(inc)
        assert res.extra["growth_rate"] > 0
        assert res.extra["doubling_time"] > 0

    def test_too_short(self):
        with pytest.raises(ValueError):
            epidemic_curve_analysis([1, 2])
