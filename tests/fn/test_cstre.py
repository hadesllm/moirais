"""Tests for moirais.fn.cstre — custody readmission rate."""

from moirais.fn.cstre import custody_readmit


class TestCustodyReadmit:
    def test_returns_dict(self, otis_df):
        result = custody_readmit(otis_df)
        assert isinstance(result, dict)

    def test_rate_bounded(self, otis_df):
        result = custody_readmit(otis_df)
        assert 0.0 <= result["readmission_rate"] <= 1.0

    def test_counts_consistent(self, otis_df):
        result = custody_readmit(otis_df)
        assert result["n_readmitted"] <= result["n_total"]

    def test_custom_cols(self, otis_df):
        df = otis_df.rename(columns={"unique_individual_id": "pid", "end_fiscal_year": "yr"})
        result = custody_readmit(df, id_col="pid", year_col="yr")
        assert result["n_total"] > 0
