"""Tests for moirais.fn.odm_g — Demographic profile per gender."""

import pandas as pd
from moirais.fn.odm_g import otis_demo_gender


class TestOtisDemoGender:
    def test_returns_dataframe(self, otis_df):
        result = otis_demo_gender(otis_df)
        assert isinstance(result, pd.DataFrame)

    def test_one_row_per_gender(self, otis_df):
        result = otis_demo_gender(otis_df)
        assert len(result) == otis_df["gender"].nunique()

    def test_has_n(self, otis_df):
        result = otis_demo_gender(otis_df)
        assert (result["n"] > 0).all()
