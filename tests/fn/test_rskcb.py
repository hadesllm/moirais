"""Tests for morie.fn.rskcb — risk calibration."""

import pandas as pd
from morie.fn.rskcb import risk_calibration, rskcb


class TestRiskCalibration:
    def test_returns_dataframe(self, otis_df):
        result = risk_calibration(otis_df)
        assert isinstance(result, pd.DataFrame)
        assert "observed_rate" in result.columns

    def test_has_bins(self, otis_df):
        result = risk_calibration(otis_df, n_bins=5)
        assert len(result) <= 5

    def test_n_sums_to_total(self, otis_df):
        result = risk_calibration(otis_df)
        assert result["n"].sum() == len(otis_df)
