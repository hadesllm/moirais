"""Tests for cfaer -- Expected parameter change."""
import pandas as pd
from moirais.fn.cfaer import cfa_expected_change
from moirais.fn._containers import DescriptiveResult


class TestCfaExpectedChange:
    def test_basic(self):
        df = pd.DataFrame({"mi": [10.5, 5.2, 1.1], "epc": [0.3, 0.1, 0.05]})
        result = cfa_expected_change(df)
        assert isinstance(result, DescriptiveResult)
        assert result.extra["n_params"] == 3

    def test_sorted_by_mi(self):
        df = pd.DataFrame({"mi": [1.0, 10.0, 5.0], "epc": [0.1, 0.3, 0.2]})
        result = cfa_expected_change(df)
        assert result.value.iloc[0]["mi"] == 10.0
