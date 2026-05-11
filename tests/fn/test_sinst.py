"""Tests for morie.fn.sinst -- Volatility of volatility."""

import numpy as np
import pandas as pd
from morie.fn.sinst import vol_of_vol, sinst
from morie.fn._containers import DescriptiveResult


class TestSinst:
    def test_alias(self):
        assert sinst is vol_of_vol

    def test_basic(self):
        rng = np.random.default_rng(42)
        df = pd.DataFrame({"x": rng.normal(0, 1, 100)})
        result = vol_of_vol(df, window=10)
        assert isinstance(result, DescriptiveResult)
        assert result.value > 0

    def test_constant_vol(self):
        x = np.sin(np.linspace(0, 20 * np.pi, 200))
        df = pd.DataFrame({"x": x})
        result = vol_of_vol(df, window=20)
        assert result.value < result.extra["mean_vol"]
