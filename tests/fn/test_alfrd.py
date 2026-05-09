"""Tests for moirais.fn.alfrd -- Cook's distance."""

import numpy as np
import pandas as pd
from moirais.fn.alfrd import cooks_distance, alfrd
from moirais.fn._containers import DescriptiveResult


class TestAlfrd:
    def test_alias(self):
        assert alfrd is cooks_distance

    def test_detects_influential(self):
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, 50)
        y = 2 * x + rng.normal(0, 0.1, 50)
        x = np.append(x, 10.0)
        y = np.append(y, -20.0)
        df = pd.DataFrame({"x": x, "y": y})
        result = cooks_distance(df, y="y", x_cols=["x"])
        assert isinstance(result, DescriptiveResult)
        assert result.value >= 1

    def test_no_influential(self):
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, 100)
        y = 2 * x + rng.normal(0, 0.1, 100)
        df = pd.DataFrame({"x": x, "y": y})
        result = cooks_distance(df, y="y")
        assert result.extra["max_cooks_d"] < 1.0
