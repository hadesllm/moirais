"""Tests for morie.fn.winros -- Wind rose plot."""

import numpy as np
import pandas as pd
from morie.fn.winros import wind_rose, winros
from morie.fn._containers import DescriptiveResult


class TestWinros:
    def test_alias(self):
        assert winros is wind_rose

    def test_basic(self):
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            "direction": rng.uniform(0, 360, 200),
            "speed": rng.exponential(5, 200),
        })
        result = wind_rose(df, n_sectors=8)
        assert isinstance(result, DescriptiveResult)
        assert len(result.value["sector_centers"]) == 8

    def test_all_north(self):
        df = pd.DataFrame({"direction": [0.0] * 50, "speed": [5.0] * 50})
        result = wind_rose(df, n_sectors=4)
        assert result.extra["dominant_direction"] == 0.0
