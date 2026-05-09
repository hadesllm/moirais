"""Tests for moirais.fn.ccf — cross-correlation function."""

import numpy as np
import pytest

from moirais.fn.ccf import ccf


class TestCcf:
    """Tests for ccf()."""

    def test_autocorrelation_at_zero(self):
        """CCF of a series with itself at lag 0 is ~1."""
        rng = np.random.default_rng(42)
        x = rng.standard_normal(100)
        result = ccf(x, x, nlags=10)
        # Lag 0 is at the center
        center = len(result["lags"]) // 2
        assert result["ccf_values"][center] == pytest.approx(1.0, abs=1e-10)

    def test_lags_symmetric(self):
        """Lag array is symmetric around 0."""
        x = np.random.default_rng(1).standard_normal(50)
        y = np.random.default_rng(2).standard_normal(50)
        result = ccf(x, y, nlags=5)
        assert result["lags"][0] == -5
        assert result["lags"][-1] == 5
        assert len(result["lags"]) == 11

    def test_ci_positive(self):
        """Confidence interval half-width is positive."""
        x = np.random.default_rng(3).standard_normal(100)
        y = np.random.default_rng(4).standard_normal(100)
        result = ccf(x, y)
        assert result["ci"] > 0
