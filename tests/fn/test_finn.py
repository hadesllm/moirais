"""Tests for morie.fn.finn -- Correlation pattern finder."""

import numpy as np
import pandas as pd
from morie.fn.finn import find_patterns, finn
from morie.fn._containers import DescriptiveResult


class TestFinn:
    def test_alias(self):
        assert finn is find_patterns

    def test_finds_correlated_columns(self):
        rng = np.random.default_rng(42)
        n = 200
        x = rng.normal(0, 1, n)
        df = pd.DataFrame({
            "x": x,
            "y": x + rng.normal(0, 0.1, n),  # highly correlated with x
            "z": rng.normal(0, 1, n),          # uncorrelated
        })
        result = find_patterns(df, threshold=0.5)
        assert isinstance(result, DescriptiveResult)
        pairs = result.extra["pairs"]
        # x-y should be found
        assert any(p["col1"] == "x" and p["col2"] == "y" for p in pairs)
        # x-z and y-z should NOT be found at threshold=0.5
        assert not any(
            (p["col1"] == "x" and p["col2"] == "z") or (p["col1"] == "y" and p["col2"] == "z")
            for p in pairs
        )

    def test_spearman(self):
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, 100)
        df = pd.DataFrame({"a": x, "b": x**3})  # monotonic but not linear
        result = find_patterns(df, method="spearman", threshold=0.3)
        assert result.extra["method"] == "spearman"
        assert len(result.extra["pairs"]) >= 1

    def test_no_pairs_below_threshold(self):
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            "a": rng.normal(0, 1, 100),
            "b": rng.normal(0, 1, 100),
        })
        result = find_patterns(df, threshold=0.99)
        assert result.value == 0
