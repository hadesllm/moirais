"""Tests for moirais.fn.cstag — custody age profile."""

import pytest
import pandas as pd
from moirais.fn.cstag import custody_age_profile


class TestCustodyAgeProfile:
    def test_returns_dataframe(self, otis_df):
        result = custody_age_profile(otis_df)
        assert isinstance(result, pd.DataFrame)

    def test_proportions_sum_to_one(self, otis_df):
        result = custody_age_profile(otis_df)
        for yr, grp in result.groupby("end_fiscal_year"):
            assert grp["proportion"].sum() == pytest.approx(1.0, abs=1e-10)

    def test_columns(self, otis_df):
        result = custody_age_profile(otis_df)
        assert "count" in result.columns
        assert "proportion" in result.columns

    def test_custom_cols(self, otis_df):
        df = otis_df.rename(columns={"age_group": "age", "end_fiscal_year": "yr"})
        result = custody_age_profile(df, age_col="age", year_col="yr")
        assert len(result) > 0
