"""Tests for morie.fn.ets — exponential smoothing (Holt-Winters)."""

import numpy as np
import pytest

from morie.fn.ets import ets


class TestEts:
    """Tests for ets()."""

    def test_additive_trend(self):
        """Additive trend produces fitted values and forecast."""
        rng = np.random.default_rng(42)
        series = np.arange(50, dtype=float) + rng.normal(0, 1, 50)
        result = ets(series, trend="add", seasonal=None)
        assert len(result["fitted"]) == 50
        assert len(result["forecast"]) == 12
        assert "alpha" in result["params"]

    def test_no_trend_no_seasonal(self):
        """Simple exponential smoothing (no trend, no seasonal)."""
        series = np.array([10, 12, 11, 13, 12, 14, 13, 15], dtype=float)
        result = ets(series, trend=None, seasonal=None)
        assert len(result["fitted"]) == len(series)
        assert result["params"]["beta"] is None

    def test_forecast_length(self):
        """Forecast has the requested number of periods."""
        series = np.sin(np.linspace(0, 4 * np.pi, 60)) + 10
        result = ets(series, trend="add", seasonal=None, forecast_periods=24)
        assert len(result["forecast"]) == 24

    def test_raises_short_series(self):
        """Too-short series raises ValueError."""
        with pytest.raises(ValueError, match="at least 3"):
            ets(np.array([1.0, 2.0]))
