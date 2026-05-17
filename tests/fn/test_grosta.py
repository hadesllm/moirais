"""Tests for morie.fn.grosta -- Grouped summary statistics."""

import numpy as np
import pandas as pd
from morie.fn.grosta import grouped_stats, grosta
from morie.fn._containers import DescriptiveResult


class TestGrosta:
    def test_alias(self):
        assert grosta is grouped_stats

    def test_correct_group_means(self):
        df = pd.DataFrame({
            "group": ["A"] * 50 + ["B"] * 50,
            "val": [10.0] * 50 + [20.0] * 50,
        })
        result = grouped_stats(df, by="group", cols=["val"])
        assert isinstance(result, DescriptiveResult)
        assert result.extra["n_groups"] == 2
        tbl = result.value
        # Check that means are 10 and 20
        means = tbl[("val", "mean")].tolist()
        assert abs(means[0] - 10.0) < 0.01
        assert abs(means[1] - 20.0) < 0.01

    def test_auto_numeric_cols(self):
        df = pd.DataFrame({
            "group": ["X", "X", "Y", "Y"],
            "a": [1.0, 2.0, 3.0, 4.0],
            "b": [10.0, 20.0, 30.0, 40.0],
            "label": ["p", "q", "r", "s"],
        })
        result = grouped_stats(df, by="group")
        # Should auto-detect numeric columns a, b (not label)
        assert "a" in result.extra["columns"]
        assert "b" in result.extra["columns"]
        assert "label" not in result.extra["columns"]
