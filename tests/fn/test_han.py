"""Tests for moirais.fn.han -- Missing data handler."""

import numpy as np
import pandas as pd
from moirais.fn.han import handle_missing, han
from moirais.fn._containers import DescriptiveResult


class TestHan:
    def test_alias(self):
        assert han is handle_missing

    def test_summary(self):
        df = pd.DataFrame({
            "a": [1.0, np.nan, 3.0],
            "b": [np.nan, np.nan, 1.0],
        })
        result = handle_missing(df, method="summary")
        assert isinstance(result, DescriptiveResult)
        assert result.value["total_missing"] == 3
        assert result.extra["per_column"]["a"] == 1
        assert result.extra["per_column"]["b"] == 2

    def test_mean_imputation(self):
        df = pd.DataFrame({"x": [1.0, np.nan, 3.0, np.nan, 5.0]})
        result = handle_missing(df, method="mean")
        imputed = result.value
        assert imputed["x"].isna().sum() == 0
        # Mean of [1, 3, 5] = 3
        assert abs(imputed["x"].iloc[1] - 3.0) < 0.01

    def test_drop(self):
        df = pd.DataFrame({
            "a": [1.0, np.nan, 3.0],
            "b": [4.0, 5.0, 6.0],
        })
        result = handle_missing(df, method="drop")
        assert result.extra["rows_before"] == 3
        assert result.extra["rows_after"] == 2
        assert result.extra["dropped"] == 1
