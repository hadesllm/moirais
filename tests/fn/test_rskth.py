"""Tests for moirais.fn.rskth — optimal risk threshold."""

from moirais.fn.rskth import risk_threshold, rskth


class TestRiskThreshold:
    def test_returns_dict(self, otis_df):
        result = risk_threshold(otis_df)
        assert isinstance(result, dict)
        assert "threshold" in result
        assert "youden_j" in result

    def test_sensitivity_specificity_bounded(self, otis_df):
        result = risk_threshold(otis_df)
        assert 0.0 <= result["sensitivity"] <= 1.0
        assert 0.0 <= result["specificity"] <= 1.0

    def test_youden_j_bounded(self, otis_df):
        result = risk_threshold(otis_df)
        assert -1.0 <= result["youden_j"] <= 1.0
