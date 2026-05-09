"""Tests for moirais.fn.cmpag — compliance by age."""

import pandas as pd
from moirais.fn.cmpag import compliance_by_age


class TestComplianceByAge:
    def test_returns_dataframe(self, otis_df):
        result = compliance_by_age(otis_df)
        assert isinstance(result, pd.DataFrame)

    def test_groups_match(self, otis_df):
        result = compliance_by_age(otis_df)
        assert set(result["age_group"]) == set(otis_df["age_group"].unique())

    def test_rate_bounded(self, otis_df):
        result = compliance_by_age(otis_df)
        assert (result["rate"] >= 0).all()
        assert (result["rate"] <= 1).all()

    def test_custom_cols(self, otis_df):
        df = otis_df.rename(columns={"D": "f", "age_group": "a"})
        result = compliance_by_age(df, flag_col="f", age_col="a")
        assert len(result) > 0
