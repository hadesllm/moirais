"""Tests for morie.fn.naifor -- Naive time series forecasting."""

import numpy as np
import pandas as pd
from morie.fn.naifor import naive_forecast, naifor
from morie.fn._containers import DescriptiveResult


class TestNaifor:
    def test_alias(self):
        assert naifor is naive_forecast

    def test_naive(self):
        df = pd.DataFrame({"x": [1, 2, 3, 4, 5]})
        result = naive_forecast(df, h=3, method="naive")
        assert isinstance(result, DescriptiveResult)
        assert all(v == 5.0 for v in result.value["forecast"])

    def test_mean(self):
        df = pd.DataFrame({"x": [2, 4, 6, 8, 10]})
        result = naive_forecast(df, h=2, method="mean")
        assert abs(result.value["forecast"][0] - 6.0) < 0.01

    def test_drift(self):
        df = pd.DataFrame({"x": np.arange(10, dtype=float)})
        result = naive_forecast(df, h=5, method="drift")
        assert result.value["forecast"][0] > 9.0
