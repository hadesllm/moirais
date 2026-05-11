"""Tests for morie.fn.prdnt — Bootstrap prediction intervals."""

import numpy as np
import pytest

from morie.fn.prdnt import prediction_intervals, prdnt


def test_returns_descriptive_result():
    """Return type has DescriptiveResult interface."""
    rng = np.random.default_rng(0)
    y = rng.standard_normal(50)
    fitted = y + rng.standard_normal(50) * 0.1
    r = prediction_intervals(y, fitted)
    assert hasattr(r, "value")
    assert hasattr(r, "extra")
    assert "lower" in r.extra and "upper" in r.extra


def test_lower_lt_upper():
    """Lower bound must be strictly less than upper bound."""
    rng = np.random.default_rng(1)
    y = rng.standard_normal(100)
    fitted = np.full(100, y.mean())
    r = prediction_intervals(y, fitted, alpha=0.05, n_boot=500)
    assert r.extra["lower"] < r.extra["upper"]


def test_forecast_in_interval():
    """The mean bootstrap forecast should lie within the PI."""
    rng = np.random.default_rng(2)
    y = rng.standard_normal(80)
    fitted = y * 0.9
    r = prediction_intervals(y, fitted, alpha=0.10, n_boot=1000)
    assert r.extra["lower"] <= r.value <= r.extra["upper"]


def test_wider_interval_for_lower_alpha():
    """95% PI should be wider than 90% PI (same data, same seed)."""
    rng = np.random.default_rng(3)
    y = rng.standard_normal(60)
    fitted = y
    r90 = prediction_intervals(y, fitted, alpha=0.10, n_boot=1000, seed=0)
    r95 = prediction_intervals(y, fitted, alpha=0.05, n_boot=1000, seed=0)
    width90 = r90.extra["upper"] - r90.extra["lower"]
    width95 = r95.extra["upper"] - r95.extra["lower"]
    assert width95 >= width90 - 1e-9


def test_residual_std_positive():
    """residual_std should be positive for non-constant residuals."""
    rng = np.random.default_rng(4)
    y = rng.standard_normal(50)
    fitted = y * 0.5
    r = prediction_intervals(y, fitted)
    assert r.extra["residual_std"] > 0.0


def test_se_nonneg():
    rng = np.random.default_rng(5)
    y = rng.standard_normal(50)
    fitted = y
    r = prediction_intervals(y, fitted, n_boot=200)
    assert r.extra["se"] >= 0.0


def test_block_method_runs():
    """Block bootstrap method runs and produces valid output."""
    rng = np.random.default_rng(6)
    y = rng.standard_normal(80)
    fitted = y * 0.8
    r = prediction_intervals(y, fitted, method="block", n_boot=300)
    assert r.extra["lower"] < r.extra["upper"]


def test_reproducibility():
    """Same seed → same output."""
    rng = np.random.default_rng(7)
    y = rng.standard_normal(60)
    fitted = y * 0.7
    r1 = prediction_intervals(y, fitted, seed=99, n_boot=500)
    r2 = prediction_intervals(y, fitted, seed=99, n_boot=500)
    assert r1.extra["lower"] == r2.extra["lower"]
    assert r1.extra["upper"] == r2.extra["upper"]


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        prediction_intervals([1.0, 2.0, 3.0], [1.0, 2.0])


def test_invalid_alpha_raises():
    with pytest.raises(ValueError):
        prediction_intervals([1.0, 2.0], [1.0, 2.0], alpha=1.5)


def test_invalid_method_raises():
    with pytest.raises(ValueError):
        prediction_intervals([1.0, 2.0], [1.0, 2.0], method="xyz")


def test_alias():
    assert prdnt is prediction_intervals
