"""Tests for moirais.fn.cstsg — custody segregation."""

from moirais.fn.cstsg import custody_segregation


class TestCustodySegregation:
    def test_returns_dict(self, otis_df):
        result = custody_segregation(otis_df)
        assert isinstance(result, dict)

    def test_proportion_bounded(self, otis_df):
        result = custody_segregation(otis_df)
        assert 0.0 <= result["proportion"] <= 1.0

    def test_counts_consistent(self, otis_df):
        result = custody_segregation(otis_df)
        assert result["n_flagged"] <= result["n_total"]

    def test_custom_cols(self, otis_df):
        df = otis_df.rename(columns={"alert_mental_health": "flag", "unique_individual_id": "pid"})
        result = custody_segregation(df, alert_col="flag", id_col="pid")
        assert result["n_total"] > 0
