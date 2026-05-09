"""Tests for moirais.fn.rey_tw — Tweedie regression."""

import numpy as np
import pandas as pd
import pytest

from moirais.fn.rey_tw import rey_tw


class TestReyTw:
    """Tests for rey_tw()."""

    def test_basic_tweedie_fit(self):
        """Fits on synthetic non-negative data."""
        rng = np.random.default_rng(42)
        n = 200
        x = rng.standard_normal(n)
        mu = np.exp(1.0 + 0.3 * x)
        y = rng.gamma(shape=2, scale=mu / 2, size=n)
        df = pd.DataFrame({"y": y, "x": x})
        result = rey_tw(df, y="y", x="x", power=1.5)
        assert result.method == "Tweedie GLM"
        assert result.n == n
        assert result.extra["power"] == 1.5

    def test_coefficient_sign(self):
        """Positive coefficient recovered."""
        rng = np.random.default_rng(7)
        n = 300
        x = rng.standard_normal(n)
        mu = np.exp(0.5 + 0.7 * x)
        y = rng.gamma(shape=3, scale=mu / 3, size=n)
        df = pd.DataFrame({"y": y, "x": x})
        result = rey_tw(df, y="y", x="x", power=1.5)
        assert result.coefficients["x"] > 0

    def test_raises_invalid_power(self):
        """Power outside (1,2) raises ValueError."""
        df = pd.DataFrame({"y": [1, 2, 3], "x": [1, 2, 3]})
        with pytest.raises(ValueError, match="power"):
            rey_tw(df, y="y", x="x", power=0.5)
