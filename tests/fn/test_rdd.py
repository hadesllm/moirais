"""Tests for moirais.fn.rdd -- Regression discontinuity design."""

import numpy as np
import pandas as pd
from moirais.fn.rdd import reg_discontinuity, rdd
from moirais.fn._containers import RegressionResult


class TestRDD:
    def test_alias(self):
        assert rdd is reg_discontinuity

    def test_recovers_jump(self):
        """Data with a jump of 5 at cutoff=0; LATE should be near 5."""
        rng = np.random.default_rng(42)
        n = 1000
        r = rng.uniform(-2, 2, n)
        y = 2 + 1.5 * r + 5 * (r >= 0).astype(float) + rng.normal(0, 0.5, n)
        df = pd.DataFrame({"outcome": y, "running": r})
        result = reg_discontinuity(df, r="running", cutoff=0.0, bandwidth=1.5)
        assert isinstance(result, RegressionResult)
        assert result.method == "Sharp RDD"
        assert abs(result.coefficients["LATE"] - 5.0) < 1.5
        assert result.p_values["LATE"] < 0.05

    def test_no_jump(self):
        """Continuous relationship, no jump; LATE should not be significant."""
        rng = np.random.default_rng(42)
        n = 500
        r = rng.uniform(-3, 3, n)
        y = 1 + 0.5 * r + rng.normal(0, 1, n)
        df = pd.DataFrame({"outcome": y, "running": r})
        result = reg_discontinuity(df, r="running", cutoff=0.0, bandwidth=2.0)
        assert abs(result.coefficients["LATE"]) < 1.5

    def test_bandwidth_in_extra(self):
        rng = np.random.default_rng(42)
        r = rng.uniform(-2, 2, 200)
        y = r + rng.normal(0, 1, 200)
        df = pd.DataFrame({"outcome": y, "running": r})
        result = reg_discontinuity(df, r="running", bandwidth=1.0)
        assert result.extra["bandwidth"] == 1.0
