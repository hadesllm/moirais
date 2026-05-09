"""Tests for moirais.fn.rey_nb — negative binomial regression."""

import numpy as np
import pandas as pd
import pytest

from moirais.fn.rey_nb import rey_nb


class TestReyNb:
    """Tests for rey_nb()."""

    def test_basic_fit(self):
        """Fits on synthetic count data."""
        rng = np.random.default_rng(42)
        n = 200
        x = rng.standard_normal(n)
        lam = np.exp(1.0 + 0.5 * x)
        y = rng.poisson(lam)
        df = pd.DataFrame({"y": y, "x": x})
        result = rey_nb(df, y="y", x="x")
        assert result.method == "Negative Binomial"
        assert result.n == n
        assert "intercept" in result.coefficients
        assert "x" in result.coefficients

    def test_positive_coef_direction(self):
        """Positive true coefficient recovered as positive."""
        rng = np.random.default_rng(7)
        n = 300
        x = rng.standard_normal(n)
        lam = np.exp(0.5 + 1.0 * x)
        y = rng.poisson(lam)
        df = pd.DataFrame({"y": y, "x": x})
        result = rey_nb(df, y="y", x="x")
        assert result.coefficients["x"] > 0

    def test_raises_missing_column(self):
        """Missing column raises ValueError."""
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        with pytest.raises(ValueError, match="not found"):
            rey_nb(df, y="y", x="a")
