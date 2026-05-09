"""Tests for moirais.fn.stprd — Spatio-temporal prediction intervals."""

import numpy as np
import pytest

from moirais.fn.stprd import stprd


class TestStprd:

    def test_prediction_output(self):
        rng = np.random.default_rng(42)
        coords = rng.uniform(0, 10, (15, 2))
        times = np.linspace(0, 5, 15)
        values = rng.standard_normal(15)
        result = stprd(coords, times, values, np.array([5.0, 5.0]), 2.5)
        assert "prediction" in result
        assert result["lower"] < result["upper"]

    def test_interval_contains_prediction(self):
        rng = np.random.default_rng(42)
        coords = rng.uniform(0, 10, (20, 2))
        times = np.linspace(0, 5, 20)
        values = rng.standard_normal(20)
        result = stprd(coords, times, values, np.array([5.0, 5.0]), 2.5)
        assert result["lower"] <= result["prediction"] <= result["upper"]

    def test_coords_mismatch_raises(self):
        with pytest.raises(ValueError, match="coords must be"):
            stprd(np.ones((5, 2)), np.ones(5), np.ones(3), np.ones(2), 1.0)
