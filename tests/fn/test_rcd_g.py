"""Tests for moirais.fn.rcd_g — recidivism by gender."""

import pandas as pd
from moirais.fn.rcd_g import recidivism_by_gender, rcd_g


class TestRecidivismByGender:
    def test_returns_dataframe(self, otis_df):
        result = recidivism_by_gender(otis_df)
        assert isinstance(result, pd.DataFrame)
        assert "gender" in result.columns

    def test_has_rate_column(self, otis_df):
        result = recidivism_by_gender(otis_df)
        assert "rate" in result.columns

    def test_two_genders(self, otis_df):
        result = recidivism_by_gender(otis_df)
        assert len(result) == 2  # Male, Female in fixture
