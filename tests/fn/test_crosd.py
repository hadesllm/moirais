"""Tests for moirais.fn.crosd — Croston's intermittent demand method."""

import numpy as np
import pytest

from moirais.fn.crosd import croston_method, crosd


def test_returns_descriptive_result():
    """Return type has DescriptiveResult interface."""
    y = np.array([0, 0, 5.0, 0, 0, 3.0, 0, 2.0])
    r = croston_method(y)
    assert hasattr(r, "value")
    assert hasattr(r, "extra")
    assert "demand_level" in r.extra


def test_all_zeros_zero_forecast():
    """Series of zeros → zero demand forecast."""
    y = np.zeros(30)
    r = croston_method(y)
    assert r.value == 0.0
    assert r.extra["combined_forecast"] == 0.0


def test_nonzero_series_positive_forecast():
    """Series with positive demands → positive forecast."""
    y = np.array([0, 3.0, 0, 0, 4.0, 0, 2.0, 0, 5.0])
    r = croston_method(y, alpha=0.2)
    assert r.value > 0.0
    assert r.extra["demand_level"] > 0.0
    assert r.extra["interval_level"] > 0.0


def test_demand_level_bounded_by_observations():
    """Smoothed demand level should be between min and max observed demands."""
    y = np.array([0, 1.0, 0, 10.0, 0, 5.0, 0, 3.0])
    r = croston_method(y, alpha=0.3)
    demands = y[y > 0]
    assert demands.min() <= r.extra["demand_level"] <= demands.max()


def test_sba_correction():
    """SBA forecast should be (1 - alpha/2) * combined_forecast."""
    y = np.array([0, 4.0, 0, 2.0, 0, 6.0, 0, 3.0])
    alpha = 0.2
    r = croston_method(y, alpha=alpha, sba=True)
    expected_sba = (1.0 - alpha / 2.0) * r.extra["combined_forecast"]
    assert abs(r.value - expected_sba) < 1e-12


def test_sba_le_combined():
    """SBA-corrected forecast <= combined forecast (bias correction reduces it)."""
    y = np.array([0, 5.0, 0, 3.0, 0, 4.0, 0, 2.0])
    r_sba = croston_method(y, alpha=0.1, sba=True)
    r_no_sba = croston_method(y, alpha=0.1, sba=False)
    assert r_sba.value <= r_no_sba.value + 1e-12


def test_n_demands_count():
    """n_demands counts non-zero entries."""
    y = np.array([0, 1.0, 0, 0, 2.0, 0, 3.0])
    r = croston_method(y)
    assert r.extra["n_demands"] == 3


def test_continuous_series():
    """Non-sparse series: all entries > 0, interval_level ≈ 1."""
    y = np.ones(20)
    r = croston_method(y, alpha=0.3)
    # All arrivals are consecutive → interval should be 1.
    assert abs(r.extra["interval_level"] - 1.0) < 0.5


def test_too_short_raises():
    with pytest.raises(ValueError):
        croston_method([5.0])


def test_invalid_alpha_raises():
    with pytest.raises(ValueError):
        croston_method([0, 1.0, 0, 2.0], alpha=1.5)


def test_alias():
    assert crosd is croston_method
