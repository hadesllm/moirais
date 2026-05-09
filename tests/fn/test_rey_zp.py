"""Tests for moirais.fn.rey_zp — zero-inflated Poisson regression."""

import numpy as np
import pandas as pd
import pytest

from moirais.fn.rey_zp import rey_zp


class TestReyZp:
    """Tests for rey_zp()."""

    def test_basic_zip_fit(self):
        """Fits on synthetic zero-inflated data."""
        rng = np.random.default_rng(42)
        n = 300
        x = rng.standard_normal(n)
        # Zero-inflation: 30% extra zeros
        is_zero = rng.binomial(1, 0.3, n)
        lam = np.exp(1.0 + 0.5 * x)
        y = np.where(is_zero, 0, rng.poisson(lam))
        df = pd.DataFrame({"y": y, "x": x})
        result = rey_zp(df, y="y", x="x")
        assert "zero_coefficients" in result
        assert "count_coefficients" in result
        assert result["n"] == n

    def test_count_coefficient_direction(self):
        """Count model recovers positive coefficient direction."""
        rng = np.random.default_rng(7)
        n = 400
        x = rng.standard_normal(n)
        is_zero = rng.binomial(1, 0.2, n)
        lam = np.exp(0.5 + 1.0 * x)
        y = np.where(is_zero, 0, rng.poisson(lam))
        df = pd.DataFrame({"y": y, "x": x})
        result = rey_zp(df, y="y", x="x")
        assert result["count_coefficients"]["x"] > 0

    def test_raises_negative_counts(self):
        """Negative counts raise ValueError."""
        df = pd.DataFrame({"y": [-1, 0, 1], "x": [1, 2, 3]})
        with pytest.raises(ValueError, match="non-negative"):
            rey_zp(df, y="y", x="x")
