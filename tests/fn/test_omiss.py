"""Tests for moirais.fn.omiss — Missing data report."""

import pandas as pd
from moirais.fn.omiss import otis_missing_report


class TestOtisMissingReport:
    def test_returns_dataframe(self, otis_df):
        result = otis_missing_report(otis_df)
        assert isinstance(result, pd.DataFrame)

    def test_one_row_per_column(self, otis_df):
        result = otis_missing_report(otis_df)
        assert len(result) == len(otis_df.columns)

    def test_no_missing_in_synthetic(self, otis_df):
        result = otis_missing_report(otis_df)
        assert (result["n_missing"] == 0).all()

    def test_sorted_by_pct(self, otis_df):
        result = otis_missing_report(otis_df)
        assert list(result["pct_missing"]) == sorted(result["pct_missing"], reverse=True)
