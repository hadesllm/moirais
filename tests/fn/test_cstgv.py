"""Tests for moirais.fn.cstgv — custody grievance rate."""

import pandas as pd
from moirais.fn.cstgv import custody_grievance_rate


class TestCustodyGrievanceRate:
    def test_returns_dataframe(self, otis_df):
        result = custody_grievance_rate(otis_df)
        assert isinstance(result, pd.DataFrame)

    def test_rate_bounded(self, otis_df):
        result = custody_grievance_rate(otis_df)
        assert (result["rate"] >= 0).all()
        assert (result["rate"] <= 1).all()

    def test_columns(self, otis_df):
        result = custody_grievance_rate(otis_df)
        assert "n_events" in result.columns
        assert "rate" in result.columns

    def test_custom_cols(self, otis_df):
        result = custody_grievance_rate(otis_df, event_col="D", group_col="region")
        assert len(result) > 0
