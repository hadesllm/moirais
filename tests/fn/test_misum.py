"""Tests for moirais.fn.misum -- MI summary table."""

import pandas as pd
from moirais.fn.misum import mi_summary


class TestMiSummary:

    def test_returns_dataframe(self):
        fits = [
            {"level": "configural", "fit": {"chi2": 100, "df": 50, "cfi": 0.95, "tli": 0.94,
             "rmsea": 0.04, "srmr": 0.05}, "delta_fit": {}, "passed": True},
            {"level": "metric", "fit": {"chi2": 110, "df": 55, "cfi": 0.94, "tli": 0.93,
             "rmsea": 0.045, "srmr": 0.055}, "delta_fit": {"delta_cfi": 0.01, "delta_rmsea": 0.005},
             "passed": True},
        ]
        result = mi_summary(fits)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2

    def test_has_expected_columns(self):
        fits = [
            {"level": "configural", "fit": {"cfi": 0.95}, "delta_fit": {}, "passed": True},
        ]
        result = mi_summary(fits)
        assert "level" in result.columns
        assert "passed" in result.columns

    def test_highest_invariance(self):
        fits = [
            {"level": "configural", "fit": {}, "delta_fit": {}, "passed": True},
            {"level": "metric", "fit": {}, "delta_fit": {}, "passed": True},
            {"level": "scalar", "fit": {}, "delta_fit": {}, "passed": False},
        ]
        result = mi_summary(fits)
        assert result.attrs["highest_invariance"] == "metric"
