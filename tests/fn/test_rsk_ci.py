"""Tests for moirais.fn.rsk_ci -- risk ratio with CI."""

import pytest
from moirais.fn.rsk_ci import risk_ratio_ci


class TestRiskRatioCI:
    def test_known_2x2(self):
        """RR for [[20, 80], [10, 90]]: p1=0.2, p2=0.1, RR=2.0."""
        result = risk_ratio_ci([[20, 80], [10, 90]])
        assert result["risk_ratio"] == pytest.approx(2.0, rel=0.01)
        assert result["ci_lower"] > 0
        assert result["ci_upper"] > result["risk_ratio"]

    def test_equal_risks(self):
        """Equal risks should give RR near 1."""
        result = risk_ratio_ci([[30, 70], [30, 70]])
        assert result["risk_ratio"] == pytest.approx(1.0, rel=0.01)

    def test_invalid_shape_raises(self):
        with pytest.raises(ValueError, match="2.*2"):
            risk_ratio_ci([[1, 2, 3], [4, 5, 6]])
