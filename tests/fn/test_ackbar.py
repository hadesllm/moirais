"""Tests for morie.fn.ackbar -- Outlier detection."""

import numpy as np
import pandas as pd
from morie.fn.ackbar import detect_outliers, ackbar
from morie.fn._containers import DescriptiveResult


class TestAckbar:
    def test_alias(self):
        assert ackbar is detect_outliers

    def test_iqr_finds_outliers(self):
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, 100)
        # Add clear outliers
        x = np.concatenate([x, [100.0, -100.0]])
        df = pd.DataFrame({"x": x})
        result = detect_outliers(df, col="x", method="iqr")
        assert isinstance(result, DescriptiveResult)
        assert result.value >= 2  # at least the two extreme outliers
        assert 100.0 in result.extra["outlier_values"]

    def test_zscore_method(self):
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, 200)
        x = np.concatenate([x, [50.0]])
        df = pd.DataFrame({"x": x})
        result = detect_outliers(df, col="x", method="zscore", threshold=3.0)
        assert result.value >= 1
        assert 50.0 in result.extra["outlier_values"]

    def test_no_outliers(self):
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})
        result = detect_outliers(df, col="x", method="iqr", threshold=3.0)
        assert result.value == 0
