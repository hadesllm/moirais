"""Tests for moirais.fn.rcd_a — recidivism by age group."""

import pandas as pd
from moirais.fn.rcd_a import recidivism_by_age, rcd_a


class TestRecidivismByAge:
    def test_returns_dataframe(self, otis_df):
        result = recidivism_by_age(otis_df)
        assert isinstance(result, pd.DataFrame)
        assert "age_group" in result.columns

    def test_has_rate_column(self, otis_df):
        result = recidivism_by_age(otis_df)
        assert "rate" in result.columns
        assert (result["rate"] >= 0).all()

    def test_n_groups(self, otis_df):
        result = recidivism_by_age(otis_df)
        assert len(result) == otis_df["age_group"].nunique()
