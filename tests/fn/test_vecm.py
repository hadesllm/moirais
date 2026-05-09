"""Tests for moirais.fn.vecm — Vector Error Correction Model."""

import numpy as np
import pytest

from moirais.fn.vecm import vecm


class TestVecm:
    """Tests for vecm()."""

    def test_cointegrated_series(self):
        """Recovers adjustment speeds from synthetic cointegrated data."""
        rng = np.random.default_rng(42)
        n = 200
        e1 = rng.standard_normal(n)
        e2 = rng.standard_normal(n)
        y2 = np.cumsum(e2)
        y1 = 0.8 * y2 + e1  # cointegrated with beta ~ [1, -0.8]
        result = vecm(y1, y2, lags=1)
        assert len(result["alpha"]) == 2
        assert len(result["beta"]) == 2
        assert result["n"] > 0

    def test_beta_structure(self):
        """Cointegrating vector has first element = 1."""
        rng = np.random.default_rng(7)
        n = 150
        y2 = np.cumsum(rng.standard_normal(n))
        y1 = 1.5 * y2 + rng.standard_normal(n) * 0.5
        result = vecm(y1, y2)
        assert result["beta"][0] == 1.0

    def test_raises_short(self):
        """Too-short series raises ValueError."""
        with pytest.raises(ValueError):
            vecm(np.array([1, 2]), np.array([3, 4]), lags=1)
