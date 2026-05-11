"""Tests for morie.fn.srd -- Sharp RDD."""

import numpy as np
import pandas as pd
from morie.fn.srd import sharp_rd, srd
from morie.fn._containers import RegressionResult


class TestSharpRD:
    def test_alias(self):
        assert srd is sharp_rd

    def test_known_jump(self):
        """Sharp RD with known discontinuity of 5."""
        rng = np.random.default_rng(42)
        n = 1000
        r = rng.uniform(-2, 2, n)
        y = 2.0 + 0.5 * r + 5.0 * (r >= 0) + rng.normal(0, 0.5, n)
        df = pd.DataFrame({"outcome": y, "running": r})
        result = sharp_rd(df, bandwidth=1.5)
        assert isinstance(result, RegressionResult)
        assert abs(result.coefficients["LATE"] - 5.0) < 2.0

    def test_triangular_kernel(self):
        rng = np.random.default_rng(42)
        n = 500
        r = rng.uniform(-1, 1, n)
        y = 3.0 * (r >= 0) + rng.normal(0, 1, n)
        df = pd.DataFrame({"outcome": y, "running": r})
        result = sharp_rd(df, kernel="triangular")
        assert result.method == "Sharp RDD"

    def test_epanechnikov_kernel(self):
        rng = np.random.default_rng(42)
        n = 500
        r = rng.uniform(-1, 1, n)
        y = 3.0 * (r >= 0) + rng.normal(0, 1, n)
        df = pd.DataFrame({"outcome": y, "running": r})
        result = sharp_rd(df, kernel="epanechnikov")
        assert result.extra["kernel"] == "epanechnikov"

    def test_auto_bandwidth(self):
        rng = np.random.default_rng(42)
        n = 500
        r = rng.uniform(-3, 3, n)
        y = 2.0 * (r >= 0) + rng.normal(0, 1, n)
        df = pd.DataFrame({"outcome": y, "running": r})
        result = sharp_rd(df)
        assert result.extra["bandwidth"] > 0
