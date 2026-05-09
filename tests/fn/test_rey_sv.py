"""Tests for moirais.fn.rey_sv — survey-weighted regression."""

import numpy as np
import pandas as pd
import pytest

from moirais.fn.rey_sv import rey_sv


class TestReySv:
    """Tests for rey_sv()."""

    def test_basic_weighted_regression(self):
        """Fits weighted OLS and returns robust SEs."""
        rng = np.random.default_rng(42)
        n = 200
        x = rng.standard_normal(n)
        y = 2.0 + 3.0 * x + rng.normal(0, 1, n)
        w = rng.uniform(0.5, 2.0, n)
        df = pd.DataFrame({"y": y, "x": x, "w": w})
        result = rey_sv(df, y="y", x="x", weights="w")
        assert result.method == "Survey-Weighted OLS"
        assert abs(result.coefficients["x"] - 3.0) < 0.5

    def test_r_squared_bounded(self):
        """R-squared is in [0, 1] for reasonable data."""
        rng = np.random.default_rng(7)
        n = 100
        x = rng.standard_normal(n)
        y = x + rng.normal(0, 0.5, n)
        w = np.ones(n)
        df = pd.DataFrame({"y": y, "x": x, "w": w})
        result = rey_sv(df, y="y", x="x", weights="w")
        assert 0 <= result.r_squared <= 1

    def test_raises_nonpositive_weights(self):
        """Non-positive weights raise ValueError."""
        df = pd.DataFrame({"y": [1, 2], "x": [3, 4], "w": [0, 1]})
        with pytest.raises(ValueError, match="weights must be"):
            rey_sv(df, y="y", x="x", weights="w")
