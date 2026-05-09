"""Tests for moirais.fn.smr -- Standardized Mortality Ratio."""

import pytest
from moirais.fn.smr import standardized_mortality_ratio


class TestSMR:
    def test_equal_obs_exp(self):
        result = standardized_mortality_ratio(50, 50.0)
        assert result["smr"] == pytest.approx(1.0)

    def test_excess_mortality(self):
        result = standardized_mortality_ratio(100, 50.0)
        assert result["smr"] == pytest.approx(2.0)
        assert result["ci_lower"] > 1.0  # significantly elevated

    def test_ci_brackets_smr(self):
        result = standardized_mortality_ratio(30, 25.0)
        assert result["ci_lower"] < result["smr"] < result["ci_upper"]

    def test_zero_observed(self):
        result = standardized_mortality_ratio(0, 10.0)
        assert result["smr"] == 0.0
        assert result["ci_lower"] == 0.0

    def test_negative_observed_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            standardized_mortality_ratio(-1, 10.0)
