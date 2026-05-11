"""Tests for morie.fn.rcdhz — hazard rate."""

import pandas as pd
from morie.fn.rcdhz import recidivism_hazard, rcdhz


class TestRecidivismHazard:
    def test_returns_dataframe(self, otis_df):
        result = recidivism_hazard(otis_df)
        assert isinstance(result, pd.DataFrame)
        assert "hazard" in result.columns

    def test_hazard_between_0_and_1(self, otis_df):
        result = recidivism_hazard(otis_df)
        assert (result["hazard"] >= 0).all()
        assert (result["hazard"] <= 1).all()

    def test_times_sorted(self, otis_df):
        result = recidivism_hazard(otis_df)
        times = result["time"].values
        assert (times[1:] >= times[:-1]).all()
