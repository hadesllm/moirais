"""Tests for morie.fn.tpsrp — police report summary."""

import pytest
import numpy as np
import pandas as pd
from morie.fn.tpsrp import tps_report_analysis
from morie.fn._containers import DescriptiveResult


@pytest.fixture()
def report_df():
    rng = np.random.default_rng(42)
    n = 200
    offenses = ["Assault", "Theft", "Break & Enter", "Fraud", "Mischief"]
    return pd.DataFrame({
        "offense_type": rng.choice(offenses, n),
        "report_date": pd.date_range("2023-01-01", periods=n, freq="D"),
    })


class TestTpsReportAnalysis:
    def test_returns_descriptive(self, report_df):
        r = tps_report_analysis(report_df)
        assert isinstance(r, DescriptiveResult)
        assert r.name == "tps_report_analysis"

    def test_counts_sum(self, report_df):
        r = tps_report_analysis(report_df)
        assert r.extra["n_offenses"] == 200

    def test_missing_col_raises(self, report_df):
        with pytest.raises(ValueError):
            tps_report_analysis(report_df, offense_col="nonexistent")
