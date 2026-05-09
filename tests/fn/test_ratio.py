"""Tests for moirais.fn.ratio — Survey ratio estimator."""

import math

import numpy as np
import pytest

from moirais.fn.ratio import ratio_estimator


def test_returns_dict_with_keys():
    """ratio_estimator returns a dict with ratio, total_estimate, se, ci keys."""
    rng = np.random.default_rng(42)
    n = 50
    x = rng.uniform(5, 15, size=n)
    y = 2.0 * x + rng.standard_normal(n) * 0.5
    w = rng.uniform(1, 5, size=n)
    result = ratio_estimator(y, x, w, X_population_total=1000.0)
    assert isinstance(result, dict)
    for key in ("ratio", "total_estimate", "se", "ci_lower", "ci_upper"):
        assert key in result


def test_ratio_is_finite():
    """Ratio and SE should be finite."""
    rng = np.random.default_rng(42)
    n = 100
    x = rng.uniform(1, 10, size=n)
    y = 1.5 * x + rng.standard_normal(n) * 0.3
    w = np.ones(n)
    result = ratio_estimator(y, x, w, X_population_total=5000.0)
    assert math.isfinite(result["ratio"])
    assert math.isfinite(result["se"])


def test_ratio_close_to_true():
    """When y = 2*x + noise, ratio should be near 2.0."""
    rng = np.random.default_rng(42)
    n = 200
    x = rng.uniform(5, 15, size=n)
    y = 2.0 * x + rng.standard_normal(n) * 0.1
    w = np.ones(n)
    result = ratio_estimator(y, x, w, X_population_total=1000.0)
    assert abs(result["ratio"] - 2.0) < 0.2


def test_negative_pop_total_raises():
    """X_population_total <= 0 should raise ValueError."""
    with pytest.raises(ValueError, match="X_population_total"):
        ratio_estimator(
            np.array([1.0]), np.array([1.0]), np.array([1.0]),
            X_population_total=-100.0,
        )


def test_length_mismatch_raises():
    """Different length inputs should raise ValueError."""
    with pytest.raises(ValueError, match="same length"):
        ratio_estimator(
            np.array([1.0, 2.0]), np.array([1.0]),
            np.array([1.0, 1.0]), X_population_total=100.0,
        )
