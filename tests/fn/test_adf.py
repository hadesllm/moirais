"""Tests for morie.fn.adf — Augmented Dickey-Fuller test."""
import numpy as np
import pytest

from morie.fn.adf import adf_test, adf


def test_stationary_series():
    """Stationary AR(1) with phi=0.5 should have low p-value."""
    rng = np.random.default_rng(42)
    n = 300
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = 0.5 * x[i - 1] + rng.standard_normal()
    result = adf_test(x)
    assert result.p_value < 0.10


def test_random_walk():
    """Random walk (non-stationary) should have higher p-value."""
    rng = np.random.default_rng(42)
    x = np.cumsum(rng.standard_normal(300))
    result = adf_test(x)
    # Non-stationary: gamma should be near zero (small negative or positive)
    assert result.statistic > -5  # Not extremely negative


def test_adf_alias():
    assert adf is adf_test


def test_too_few_obs():
    with pytest.raises(ValueError):
        adf_test(np.ones(5))
