"""Tests for morie.fn.hales -- health-adjusted life expectancy."""

import numpy as np
import pytest
from morie.fn.hales import health_adjusted_life_expectancy


class TestHALE:
    def test_perfect_health(self):
        ages = np.array([0.0, 20.0, 40.0, 60.0])
        le = np.array([80.0, 60.0, 40.0, 20.0])
        prev = np.array([0.0, 0.0, 0.0, 0.0])
        dw = np.array([0.0, 0.0, 0.0, 0.0])
        res = health_adjusted_life_expectancy(ages, le, prev, dw)
        np.testing.assert_allclose(res["hale"], le)
        assert res["hale_at_birth"] == pytest.approx(80.0)

    def test_some_disability(self):
        ages = np.array([0.0, 40.0])
        le = np.array([80.0, 40.0])
        prev = np.array([0.1, 0.3])
        dw = np.array([0.5, 0.5])
        res = health_adjusted_life_expectancy(ages, le, prev, dw)
        assert res["hale"][0] == pytest.approx(80.0 * (1 - 0.05))
        assert res["hale"][1] == pytest.approx(40.0 * (1 - 0.15))

    def test_hale_less_than_le(self):
        ages = np.array([0.0])
        le = np.array([75.0])
        prev = np.array([0.2])
        dw = np.array([0.3])
        res = health_adjusted_life_expectancy(ages, le, prev, dw)
        assert res["hale_at_birth"] < res["le_at_birth"]

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            health_adjusted_life_expectancy(
                np.array([0, 20]), np.array([80]), np.array([0.1, 0.1]), np.array([0.5, 0.5])
            )

    def test_invalid_prevalence_raises(self):
        with pytest.raises(ValueError):
            health_adjusted_life_expectancy(
                np.array([0]), np.array([80]), np.array([1.5]), np.array([0.5])
            )
