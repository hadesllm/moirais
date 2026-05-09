"""Tests for moirais.fn.or_ci -- odds ratio with confidence interval."""

import numpy as np
import pytest
from moirais.fn.or_ci import odds_ratio_ci


class TestOddsRatioCI:
    def test_known_2x2(self):
        """OR for [[10, 5], [3, 12]] = (10*12)/(5*3) = 8.0."""
        result = odds_ratio_ci([[10, 5], [3, 12]])
        assert isinstance(result, dict)
        assert result["odds_ratio"] == pytest.approx(8.0, rel=0.01)
        assert result["ci_lower"] < result["odds_ratio"]
        assert result["ci_upper"] > result["odds_ratio"]
        assert 0 < result["p_value"] < 1

    def test_null_effect(self):
        """Equal cell counts should give OR near 1."""
        result = odds_ratio_ci([[50, 50], [50, 50]])
        assert result["odds_ratio"] == pytest.approx(1.0, rel=0.01)
        assert result["ci_lower"] < 1.0
        assert result["ci_upper"] > 1.0

    def test_invalid_shape_raises(self):
        with pytest.raises(ValueError, match="2.*2"):
            odds_ratio_ci([[1, 2, 3], [4, 5, 6]])
