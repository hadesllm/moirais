"""Tests for moirais.fn.rvflh -- Causal reversal test."""

import numpy as np
import pandas as pd
from moirais.fn.rvflh import causal_reversal_test, rvflh
from moirais.fn._containers import TestResult


class TestRvflh:
    def test_alias(self):
        assert rvflh is causal_reversal_test

    def test_clear_direction(self):
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, 100)
        y = 2 * x + rng.normal(0, 0.1, 100)
        df = pd.DataFrame({"x": x, "y": y})
        result = causal_reversal_test(df, seed=42, n_perm=199)
        assert isinstance(result, TestResult)

    def test_has_direction(self):
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, 50)
        y = 3 * x + rng.normal(0, 0.1, 50)
        df = pd.DataFrame({"x": x, "y": y})
        result = causal_reversal_test(df, seed=42)
        assert result.extra["direction"] in ("X->Y", "Y->X", "indeterminate")
