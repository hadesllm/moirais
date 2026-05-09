"""Tests for moirais.fn.dor -- Diagnostic Odds Ratio."""

import math
import pytest
from moirais.fn.dor import diagnostic_odds_ratio


class TestDiagnosticOddsRatio:
    def test_basic(self):
        result = diagnostic_odds_ratio(tp=80, fp=10, fn=20, tn=90)
        assert result["dor"] > 1
        assert result["ci_lower"] < result["dor"]
        assert result["ci_upper"] > result["dor"]

    def test_perfect_test(self):
        """Zero FP and FN (with continuity correction)."""
        result = diagnostic_odds_ratio(tp=50, fp=0, fn=0, tn=50)
        assert result["dor"] > 100  # very high

    def test_log_dor(self):
        result = diagnostic_odds_ratio(tp=80, fp=10, fn=20, tn=90)
        assert result["log_dor"] == pytest.approx(math.log(result["dor"]), abs=1e-10)

    def test_negative_raises(self):
        with pytest.raises(ValueError):
            diagnostic_odds_ratio(tp=-1, fp=10, fn=20, tn=90)
