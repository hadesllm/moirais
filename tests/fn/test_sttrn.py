"""Tests for morie.fn.sttrn — Spatio-temporal trend surface."""

import numpy as np
import pytest

from morie.fn.sttrn import sttrn


class TestSttrn:

    def test_linear_trend(self):
        rng = np.random.default_rng(42)
        n = 30
        coords = rng.uniform(0, 10, (n, 2))
        times = rng.uniform(0, 5, n)
        values = 2 * coords[:, 0] + 3 * times + rng.standard_normal(n) * 0.1
        result = sttrn(coords, times, values)
        assert result["r_squared"] > 0.8

    def test_output_keys(self):
        rng = np.random.default_rng(42)
        coords = rng.uniform(0, 10, (20, 2))
        times = rng.uniform(0, 5, 20)
        values = rng.standard_normal(20)
        result = sttrn(coords, times, values)
        assert "coefficients" in result
        assert "fitted" in result
        assert "residuals" in result
        assert result["n"] == 20

    def test_coords_mismatch_raises(self):
        with pytest.raises(ValueError, match="coords must be"):
            sttrn(np.ones((5, 3)), np.ones(5), np.ones(5))
