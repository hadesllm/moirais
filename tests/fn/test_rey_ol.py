"""Tests for morie.fn.rey_ol — ordinal logistic regression."""

import numpy as np
import pandas as pd
import pytest

from morie.fn.rey_ol import rey_ol


class TestReyOl:
    """Tests for rey_ol()."""

    def test_basic_ordinal_fit(self):
        """Fits on synthetic ordered categorical data."""
        rng = np.random.default_rng(42)
        n = 300
        x = rng.standard_normal(n)
        # Latent variable model
        z = 0.8 * x + rng.logistic(size=n)
        y = np.digitize(z, bins=[-1, 0.5, 1.5])  # 4 categories: 0,1,2,3
        df = pd.DataFrame({"y": y, "x": x})
        result = rey_ol(df, y="y", x="x")
        assert "thresholds" in result
        assert "coefficients" in result
        assert result["n"] == n
        assert len(result["categories"]) >= 2

    def test_coefficient_direction(self):
        """Positive latent slope gives positive coefficient."""
        rng = np.random.default_rng(7)
        n = 500
        x = rng.standard_normal(n)
        z = 1.5 * x + rng.logistic(size=n)
        y = np.digitize(z, bins=[-1, 0, 1])
        df = pd.DataFrame({"y": y, "x": x})
        result = rey_ol(df, y="y", x="x")
        assert result["coefficients"]["x"] > 0

    def test_raises_single_category(self):
        """Single category raises ValueError."""
        df = pd.DataFrame({"y": [1, 1, 1], "x": [1, 2, 3]})
        with pytest.raises(ValueError, match="2 ordered"):
            rey_ol(df, y="y", x="x")
