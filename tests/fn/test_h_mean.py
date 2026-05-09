"""Tests for moirais.fn.h_mean — Hajek mean estimator."""

import math

import numpy as np
import pytest

from moirais.fn.h_mean import hajek_mean


def test_returns_dict_with_keys():
    """hajek_mean returns a dict with mean, se, ci_lower, ci_upper."""
    rng = np.random.default_rng(42)
    y = rng.uniform(5, 15, size=50)
    w = rng.uniform(1, 10, size=50)
    result = hajek_mean(y, w)
    assert isinstance(result, dict)
    for key in ("mean", "se", "ci_lower", "ci_upper"):
        assert key in result


def test_mean_between_min_and_max():
    """Hajek mean should lie between min(y) and max(y)."""
    rng = np.random.default_rng(42)
    y = rng.uniform(2, 20, size=100)
    w = rng.uniform(1, 5, size=100)
    result = hajek_mean(y, w)
    assert min(y) <= result["mean"] <= max(y)


def test_equal_weights_gives_sample_mean():
    """With equal weights, Hajek mean should equal the arithmetic mean."""
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    w = np.ones(5)
    result = hajek_mean(y, w)
    assert result["mean"] == pytest.approx(np.mean(y), abs=1e-10)


def test_se_finite_and_positive():
    """SE should be finite and positive for non-degenerate data."""
    rng = np.random.default_rng(42)
    y = rng.standard_normal(50)
    w = rng.uniform(1, 10, size=50)
    result = hajek_mean(y, w)
    assert math.isfinite(result["se"])
    assert result["se"] > 0


def test_negative_weights_raises():
    """Weights <= 0 should raise ValueError."""
    with pytest.raises(ValueError, match="weights must be > 0"):
        hajek_mean(np.array([1.0, 2.0, 3.0]), np.array([1.0, -1.0, 1.0]))
