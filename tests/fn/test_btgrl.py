"""Tests for moirais.fn.btgrl -- Bagplot bivariate outliers."""

import numpy as np
import pandas as pd
from moirais.fn.btgrl import bagplot_outliers, btgrl
from moirais.fn._containers import DescriptiveResult


class TestBtgrl:
    def test_alias(self):
        assert btgrl is bagplot_outliers

    def test_detects_outlier(self):
        rng = np.random.default_rng(42)
        x = np.concatenate([rng.normal(0, 1, 50), [20.0]])
        y = np.concatenate([rng.normal(0, 1, 50), [20.0]])
        df = pd.DataFrame({"x": x, "y": y})
        result = bagplot_outliers(df, x="x", y="y")
        assert isinstance(result, DescriptiveResult)
        assert result.value >= 1

    def test_no_outliers(self):
        rng = np.random.default_rng(42)
        df = pd.DataFrame({"x": rng.normal(0, 0.1, 20), "y": rng.normal(0, 0.1, 20)})
        result = bagplot_outliers(df, x="x", y="y", factor=10.0)
        assert result.value == 0
