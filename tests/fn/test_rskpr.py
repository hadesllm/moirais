"""Tests for moirais.fn.rskpr — risk profile by subgroups."""

import pandas as pd
from moirais.fn.rskpr import risk_profile, rskpr


class TestRiskProfile:
    def test_returns_dataframe(self, otis_df):
        result = risk_profile(otis_df)
        assert isinstance(result, pd.DataFrame)
        assert "mean_score" in result.columns

    def test_custom_group(self, otis_df):
        result = risk_profile(otis_df, group_cols=["gender"])
        assert len(result) == otis_df["gender"].nunique()

    def test_has_n_column(self, otis_df):
        result = risk_profile(otis_df, group_cols=["region"])
        assert "n" in result.columns
        assert result["n"].sum() == len(otis_df)
