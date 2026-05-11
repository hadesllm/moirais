"""Tests for morie.fn.rskau — AUC for risk score."""

from morie.fn.rskau import risk_auc, rskau


class TestRiskAUC:
    def test_returns_dict(self, otis_df):
        result = risk_auc(otis_df)
        assert isinstance(result, dict)
        assert "auc" in result

    def test_auc_between_0_and_1(self, otis_df):
        result = risk_auc(otis_df)
        assert 0.0 <= result["auc"] <= 1.0

    def test_ci_contains_auc(self, otis_df):
        result = risk_auc(otis_df)
        assert result["ci_lower"] <= result["auc"] <= result["ci_upper"]
