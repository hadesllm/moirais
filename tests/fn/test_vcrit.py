"""Tests for morie.fn.vcrit — Criterion validity."""

import numpy as np
import pytest
from morie.fn.vcrit import validity_criterion


class TestValidityCriterion:

    def test_high_correlation(self, rng):
        x = rng.standard_normal(100)
        y = x + rng.standard_normal(100) * 0.2
        result = validity_criterion(x, y)
        assert result["r"] > 0.8
        assert result["p_value"] < 0.05

    def test_ci_contains_r(self, rng):
        x = rng.standard_normal(100)
        y = x + rng.standard_normal(100) * 0.5
        result = validity_criterion(x, y)
        assert result["ci_lower"] <= result["r"] <= result["ci_upper"]

    def test_small_n(self):
        result = validity_criterion(np.array([1.0, 2.0]), np.array([3.0, 4.0]))
        assert np.isnan(result["r"])

    def test_n_correct(self, rng):
        x = rng.standard_normal(50)
        result = validity_criterion(x, x)
        assert result["n"] == 50

    def test_handles_nan(self, rng):
        x = rng.standard_normal(50)
        y = x.copy()
        y[0] = np.nan
        result = validity_criterion(x, y)
        assert result["n"] == 49
