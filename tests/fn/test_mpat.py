"""Tests for moirais.fn.mpat — missing data pattern analysis."""
import numpy as np
import pytest
from moirais.fn.mpat import missing_pattern


class TestMissingPattern:
    def test_basic_pattern(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((100, 4))
        X[0:10, 0] = np.nan
        X[20:40, 2] = np.nan
        res = missing_pattern(X)
        assert res.extra["n_complete_cases"] < 100
        assert 0.0 < res.extra["pct_missing_overall"] < 1.0

    def test_no_missing(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((50, 3))
        res = missing_pattern(X)
        assert res.extra["n_complete_cases"] == 50
        assert res.extra["pct_missing_overall"] == 0.0

    def test_per_col_length(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((80, 5))
        X[:, 1] = np.nan
        res = missing_pattern(X)
        assert len(res.extra["pct_missing_per_col"]) == 5
        assert res.extra["pct_missing_per_col"][1] == 1.0
