"""Tests for moirais.fn.rskdc — outcome rate by risk decile."""

import pandas as pd
from moirais.fn.rskdc import risk_decile, rskdc


class TestRiskDecile:
    def test_returns_dataframe(self, otis_df):
        result = risk_decile(otis_df)
        assert isinstance(result, pd.DataFrame)
        assert "outcome_rate" in result.columns

    def test_has_decile_column(self, otis_df):
        result = risk_decile(otis_df)
        assert "decile" in result.columns
        assert len(result) <= 10

    def test_n_sums_to_total(self, otis_df):
        result = risk_decile(otis_df)
        assert result["n"].sum() == len(otis_df)
