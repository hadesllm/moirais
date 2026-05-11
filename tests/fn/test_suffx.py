"""Tests for morie.fn.suffx — sufficient statistic test."""

import numpy as np
import pytest

from morie.fn.suffx import suffx


class TestSuffx:
    def test_runs_and_returns_keys(self):
        rng = np.random.default_rng(42)
        data = rng.normal(5.0, 1.0, size=50)

        def stat(x):
            return float(x)

        def ll(x, theta):
            return -0.5 * (x - theta) ** 2

        result = suffx(data, stat, ll, np.linspace(3, 7, 5))
        assert "is_sufficient" in result
        assert "max_violation" in result
        assert result["max_violation"] >= 0

    def test_output_keys(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = suffx(
            data,
            lambda x: float(x),
            lambda x, t: -0.5 * (x - t) ** 2,
            np.array([0.0, 1.0, 2.0]),
        )
        assert "is_sufficient" in result
        assert "max_violation" in result
        assert "n_comparisons" in result

    def test_max_violation_nonnegative(self):
        data = np.array([1.0, 2.0, 3.0])
        result = suffx(
            data,
            lambda x: float(x),
            lambda x, t: -0.5 * (x - t) ** 2,
            np.array([0.0, 1.0]),
        )
        assert result["max_violation"] >= 0
