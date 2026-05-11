"""Tests for morie.fn.rskbr — risk base rate."""

import pandas as pd
from morie.fn.rskbr import risk_base_rate, rskbr


class TestRiskBaseRate:
    def test_returns_dict_no_group(self, otis_df):
        result = risk_base_rate(otis_df)
        assert isinstance(result, dict)
        assert "base_rate" in result
        assert 0.0 <= result["base_rate"] <= 1.0

    def test_returns_dataframe_with_group(self, otis_df):
        result = risk_base_rate(otis_df, group_col="gender")
        assert isinstance(result, pd.DataFrame)
        assert "base_rate" in result.columns

    def test_n_total_matches(self, otis_df):
        result = risk_base_rate(otis_df)
        assert result["n_total"] == len(otis_df)
