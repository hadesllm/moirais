"""Tests for morie.fn.winmea -- Winsorized mean."""

import numpy as np
import pandas as pd
from morie.fn.winmea import winsorized_mean, winmea
from morie.fn._containers import DescriptiveResult


class TestWinmea:
    def test_alias(self):
        assert winmea is winsorized_mean

    def test_no_trim(self):
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})
        result = winsorized_mean(df, col="x", proportion=0.0)
        assert isinstance(result, DescriptiveResult)
        assert abs(result.value - 3.0) < 1e-10

    def test_trims_outliers(self):
        x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 100]
        df = pd.DataFrame({"x": x})
        result = winsorized_mean(df, col="x", proportion=0.1)
        assert result.value < np.mean(x)
