"""Tests for moirais.fn.obund — OTIS Manski bounds."""

import pytest
from moirais.fn.obund import otis_bounds
from moirais.fn._containers import ESRes


class TestOtisBounds:

    def test_returns_esres(self):
        result = otis_bounds(0, 1, 0.5, mean_treated=0.7, mean_control=0.3)
        assert isinstance(result, ESRes)
        assert result.ci_lower <= result.ci_upper

    def test_bounds_contain_midpoint(self):
        result = otis_bounds(0, 1, 0.5, mean_treated=0.6, mean_control=0.4)
        assert result.ci_lower <= result.estimate <= result.ci_upper
