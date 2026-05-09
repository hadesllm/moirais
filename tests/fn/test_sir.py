"""Tests for moirais.fn.sir -- Standardized Incidence Ratio."""

import pytest
from moirais.fn.sir import standardized_incidence_ratio


class TestSIR:
    def test_equal_obs_exp(self):
        result = standardized_incidence_ratio(40, 40.0)
        assert result["sir"] == pytest.approx(1.0)

    def test_elevated_incidence(self):
        result = standardized_incidence_ratio(80, 40.0)
        assert result["sir"] == pytest.approx(2.0)

    def test_ci_brackets_sir(self):
        result = standardized_incidence_ratio(25, 20.0)
        assert result["ci_lower"] < result["sir"] < result["ci_upper"]

    def test_negative_observed_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            standardized_incidence_ratio(-5, 10.0)
