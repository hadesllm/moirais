"""Tests for moirais.fn.rey_gm — Gamma GLM regression."""

import numpy as np
import pandas as pd
import pytest

from moirais.fn.rey_gm import rey_gm


class TestReyGm:
    """Tests for rey_gm()."""

    def test_basic_gamma_fit(self):
        """Fits on synthetic Gamma-distributed data."""
        rng = np.random.default_rng(42)
        n = 200
        x = rng.standard_normal(n)
        mu = np.exp(1.0 + 0.5 * x)
        y = rng.gamma(shape=5, scale=mu / 5, size=n)
        df = pd.DataFrame({"y": y, "x": x})
        result = rey_gm(df, y="y", x="x")
        assert result.method == "Gamma GLM"
        assert result.n == n
        assert "intercept" in result.coefficients

    def test_coefficient_direction(self):
        """Positive covariate effect recovered."""
        rng = np.random.default_rng(7)
        n = 300
        x = rng.standard_normal(n)
        mu = np.exp(2.0 + 0.8 * x)
        y = rng.gamma(shape=10, scale=mu / 10, size=n)
        df = pd.DataFrame({"y": y, "x": x})
        result = rey_gm(df, y="y", x="x")
        assert result.coefficients["x"] > 0

    def test_raises_nonpositive_response(self):
        """Non-positive response raises ValueError."""
        df = pd.DataFrame({"y": [0, 1, 2], "x": [1, 2, 3]})
        with pytest.raises(ValueError, match="positive"):
            rey_gm(df, y="y", x="x")
