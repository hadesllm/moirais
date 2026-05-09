"""Tests for moirais.fn.prev -- Point prevalence."""

import pytest
from moirais.fn.prev import point_prevalence


class TestPointPrevalence:
    def test_known_prevalence(self):
        result = point_prevalence(25, 100)
        assert result["prevalence"] == pytest.approx(0.25)

    def test_ci_in_zero_one(self):
        result = point_prevalence(10, 200)
        assert 0 <= result["ci_lower"] <= result["ci_upper"] <= 1

    def test_zero_cases(self):
        result = point_prevalence(0, 100)
        assert result["prevalence"] == 0.0
        assert result["ci_lower"] >= 0.0

    def test_cases_exceed_total_raises(self):
        with pytest.raises(ValueError, match="exceed"):
            point_prevalence(101, 100)
