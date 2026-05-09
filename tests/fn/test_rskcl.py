"""Tests for moirais.fn.rskcl — risk classification."""

import pandas as pd
from moirais.fn.rskcl import risk_classify, rskcl


class TestRiskClassify:
    def test_returns_dataframe_with_risk_level(self, otis_df):
        result = risk_classify(otis_df)
        assert isinstance(result, pd.DataFrame)
        assert "risk_level" in result.columns

    def test_three_levels(self, otis_df):
        result = risk_classify(otis_df)
        levels = set(result["risk_level"].unique())
        assert levels == {"Low", "Medium", "High"}

    def test_same_length(self, otis_df):
        result = risk_classify(otis_df)
        assert len(result) == len(otis_df)
