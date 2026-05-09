"""Tests for fn/hajek.py -- Hajek estimator for population mean."""
import numpy as np
import pytest

from moirais.fn.hajek import hajek, hajek_mean


def test_hajek_returns_dict():
    rng = np.random.default_rng(42)
    y = rng.normal(50, 10, size=100)
    w = rng.uniform(1, 5, size=100)
    result = hajek(y, w)
    assert isinstance(result, dict)
    assert "mean" in result
    assert "se" in result
    assert "ci_lower" in result
    assert "ci_upper" in result


def test_hajek_equal_weights():
    """With equal weights, Hajek mean should equal arithmetic mean."""
    rng = np.random.default_rng(42)
    y = rng.normal(10, 2, size=200)
    w = np.ones(200)
    result = hajek_mean(y, w)
    assert abs(result["mean"] - np.mean(y)) < 1e-10


def test_hajek_ci_contains_mean():
    rng = np.random.default_rng(42)
    y = rng.normal(100, 5, size=500)
    w = rng.uniform(1, 3, size=500)
    result = hajek(y, w)
    assert result["ci_lower"] < result["mean"] < result["ci_upper"]


def test_hajek_bad_weights():
    y = np.array([1.0, 2.0, 3.0])
    w = np.array([1.0, -1.0, 1.0])
    with pytest.raises(ValueError):
        hajek(y, w)
