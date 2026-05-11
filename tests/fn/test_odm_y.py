"""Tests for morie.fn.odm_y — Demographic profile per year."""

import pandas as pd
from morie.fn.odm_y import otis_demo_year


class TestOtisDemoYear:
    def test_returns_dataframe(self, otis_df):
        result = otis_demo_year(otis_df)
        assert isinstance(result, pd.DataFrame)

    def test_one_row_per_year(self, otis_df):
        result = otis_demo_year(otis_df)
        assert len(result) == otis_df["end_fiscal_year"].nunique()

    def test_has_n(self, otis_df):
        result = otis_demo_year(otis_df)
        assert (result["n"] > 0).all()
