"""Tests for morie.fn.windu -- Wald test ('This party is over.')."""

import pytest
from morie.fn.windu import wald_test, windu
from morie.fn._containers import TestResult


class TestWindu:
    def test_alias(self):
        assert windu is wald_test

    def test_significant_result(self):
        """estimate=2, se=0.5, null=0 => z=4, should strongly reject."""
        result = wald_test(2.0, 0.5, null=0.0)
        assert isinstance(result, TestResult)
        assert result.test_name == "Wald"
        assert abs(result.statistic - 4.0) < 1e-10
        assert result.p_value < 0.001

    def test_non_significant_result(self):
        """estimate=0.1, se=1.0, null=0 => z=0.1, should not reject."""
        result = wald_test(0.1, 1.0)
        assert result.p_value > 0.05

    def test_custom_null(self):
        """Test with non-zero null hypothesis."""
        result = wald_test(5.0, 0.5, null=5.0)
        assert abs(result.statistic) < 1e-10
        assert result.p_value > 0.99

    def test_negative_se_raises(self):
        with pytest.raises(ValueError, match="positive"):
            wald_test(1.0, -0.5)

    def test_zero_se_raises(self):
        with pytest.raises(ValueError, match="positive"):
            wald_test(1.0, 0.0)
