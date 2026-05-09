"""Tests for moirais.fn.leia -- Dataset profiling."""

import numpy as np
import pandas as pd
from moirais.fn.leia import profile, leia
from moirais.fn._containers import DescriptiveResult


class TestLeia:
    def test_alias(self):
        assert leia is profile

    def test_shape_reported(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": ["x", "y", "z"]})
        result = profile(df)
        assert isinstance(result, DescriptiveResult)
        assert result.value["n_rows"] == 3
        assert result.value["n_cols"] == 3

    def test_missing_detected(self):
        df = pd.DataFrame({"x": [1.0, np.nan, 3.0], "y": [np.nan, np.nan, 1.0]})
        result = profile(df)
        assert result.extra["missing"]["x"] == 1
        assert result.extra["missing"]["y"] == 2
        assert result.extra["pct_missing"]["y"] > 50

    def test_high_correlation_found(self):
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, 100)
        df = pd.DataFrame({"a": x, "b": x * 0.95 + rng.normal(0, 0.1, 100), "c": rng.normal(0, 1, 100)})
        result = profile(df)
        corrs = result.extra["high_correlations"]
        # a and b should be highly correlated
        assert any(p["col1"] == "a" and p["col2"] == "b" for p in corrs)
