"""Tests for morie.fn.cmpgn — compliance by gender."""

import pandas as pd
from morie.fn.cmpgn import compliance_by_gender


class TestComplianceByGender:
    def test_returns_dataframe(self, otis_df):
        result = compliance_by_gender(otis_df)
        assert isinstance(result, pd.DataFrame)

    def test_groups_match(self, otis_df):
        result = compliance_by_gender(otis_df)
        assert set(result["gender"]) == set(otis_df["gender"].unique())

    def test_rate_bounded(self, otis_df):
        result = compliance_by_gender(otis_df)
        assert (result["rate"] >= 0).all()
        assert (result["rate"] <= 1).all()

    def test_custom_cols(self, otis_df):
        df = otis_df.rename(columns={"D": "f", "gender": "g"})
        result = compliance_by_gender(df, flag_col="f", gender_col="g")
        assert len(result) > 0
