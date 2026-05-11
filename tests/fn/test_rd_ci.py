"""Tests for morie.fn.rd_ci -- risk difference with confidence interval."""

import pytest
from morie.fn.rd_ci import risk_difference_ci


class TestRiskDifferenceCI:
    def test_known_2x2(self):
        """RD for [[20, 80], [10, 90]]: p1=0.2, p2=0.1, RD=0.1."""
        result = risk_difference_ci([[20, 80], [10, 90]])
        assert isinstance(result, dict)
        assert result["risk_difference"] == pytest.approx(0.1, abs=0.001)
        assert result["ci_lower"] < result["risk_difference"]
        assert result["ci_upper"] > result["risk_difference"]

    def test_equal_proportions(self):
        """Equal proportions should give RD near 0."""
        result = risk_difference_ci([[30, 70], [30, 70]])
        assert result["risk_difference"] == pytest.approx(0.0, abs=0.001)

    def test_invalid_shape_raises(self):
        with pytest.raises(ValueError, match="2.*2"):
            risk_difference_ci([[1, 2, 3]])
