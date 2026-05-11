"""Tests for morie.fn.sace -- SACE estimator."""

import numpy as np
import pandas as pd
import pytest
from morie.fn.sace import sace as sace_fn
from morie.fn._containers import ESRes


class TestSACE:
    def test_known_effect(self):
        """SACE with monotonicity: treatment helps survival and outcome."""
        rng = np.random.default_rng(42)
        n = 1000
        t = rng.binomial(1, 0.5, n).astype(float)
        s = np.where(t == 1, rng.binomial(1, 0.9, n), rng.binomial(1, 0.7, n)).astype(float)
        y = np.where(s == 1, 5.0 + 2.0 * t + rng.normal(0, 1, n), np.nan)
        df = pd.DataFrame({"outcome": y, "treatment": t, "survival": s})
        result = sace_fn(df)
        assert isinstance(result, ESRes)
        assert result.measure == "SACE (trimming)"
        assert abs(result.estimate - 2.0) < 1.5
        assert result.se > 0

    def test_no_survivors_raises(self):
        df = pd.DataFrame({"outcome": [1.0], "treatment": [1.0], "survival": [0.0]})
        with pytest.raises(ValueError, match="No survivors"):
            sace_fn(df)

    def test_ci_contains_estimate(self):
        rng = np.random.default_rng(99)
        n = 500
        t = rng.binomial(1, 0.5, n).astype(float)
        s = np.where(t == 1, rng.binomial(1, 0.85, n), rng.binomial(1, 0.8, n)).astype(float)
        y = np.where(s == 1, 3.0 + 1.0 * t + rng.normal(0, 0.5, n), np.nan)
        df = pd.DataFrame({"outcome": y, "treatment": t, "survival": s})
        result = sace_fn(df)
        assert result.ci_lower <= result.estimate <= result.ci_upper
