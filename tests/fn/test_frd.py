"""Tests for moirais.fn.frd -- Fuzzy RDD."""

import numpy as np
import pandas as pd
import pytest
from moirais.fn.frd import fuzzy_rd, frd
from moirais.fn._containers import RegressionResult


class TestFuzzyRD:
    def test_alias(self):
        assert frd is fuzzy_rd

    def test_known_late(self):
        """Fuzzy RD with known LATE."""
        rng = np.random.default_rng(42)
        n = 1000
        r = rng.uniform(-2, 2, n)
        prob_t = np.where(r >= 0, 0.8, 0.2)
        t = rng.binomial(1, prob_t, n).astype(float)
        y = 1.0 + 3.0 * t + 0.5 * r + rng.normal(0, 1, n)
        df = pd.DataFrame({"outcome": y, "treatment": t, "running": r})
        result = fuzzy_rd(df)
        assert isinstance(result, RegressionResult)
        assert "LATE" in result.coefficients
        assert abs(result.coefficients["LATE"] - 3.0) < 3.0

    def test_first_stage_jump(self):
        rng = np.random.default_rng(42)
        n = 500
        r = rng.uniform(-1, 1, n)
        t = np.where(r >= 0, rng.binomial(1, 0.9, n), rng.binomial(1, 0.1, n)).astype(float)
        y = 2.0 * t + rng.normal(0, 1, n)
        df = pd.DataFrame({"outcome": y, "treatment": t, "running": r})
        result = fuzzy_rd(df)
        assert abs(result.extra["first_stage_jump"]) > 0.3

    def test_too_few_obs_raises(self):
        df = pd.DataFrame({"outcome": [1, 2], "treatment": [0, 1], "running": [0, 0.01]})
        with pytest.raises(ValueError, match="Too few"):
            fuzzy_rd(df, bandwidth=0.001)
