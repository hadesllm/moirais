"""Tests for moirais.fn.rskfr — risk fairness."""

import pandas as pd
from moirais.fn.rskfr import risk_fairness, rskfr


class TestRiskFairness:
    def test_returns_dataframe(self, otis_df):
        result = risk_fairness(otis_df)
        assert isinstance(result, pd.DataFrame)
        assert "fpr" in result.columns
        assert "fnr" in result.columns

    def test_one_row_per_group(self, otis_df):
        result = risk_fairness(otis_df)
        assert len(result) == otis_df["gender"].nunique()

    def test_rates_bounded(self, otis_df):
        result = risk_fairness(otis_df)
        assert (result["fpr"].dropna() >= 0).all()
        assert (result["fpr"].dropna() <= 1).all()
