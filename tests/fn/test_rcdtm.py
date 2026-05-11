"""Tests for morie.fn.rcdtm — recidivism time-to-event summary."""

import numpy as np
from morie.fn.rcdtm import recidivism_time, rcdtm


class TestRecidivismTime:
    def test_returns_dict(self, otis_df):
        result = recidivism_time(otis_df)
        assert isinstance(result, dict)
        assert "mean" in result
        assert "median" in result

    def test_n_total(self, otis_df):
        result = recidivism_time(otis_df)
        assert result["n_total"] == len(otis_df)

    def test_events_plus_censored(self, otis_df):
        result = recidivism_time(otis_df)
        assert result["n_events"] + result["n_censored"] == result["n_total"]
