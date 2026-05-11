"""Tests for morie.fn.ljung — Ljung-Box test."""
import numpy as np

from morie.fn.ljung import ljung_box, ljung


def test_white_noise_not_rejected():
    """White noise should not reject (p > 0.05)."""
    rng = np.random.default_rng(42)
    x = rng.standard_normal(200)
    result = ljung_box(x, n_lags=10)
    assert result.p_value > 0.05


def test_autocorrelated_rejected():
    """AR(1) process should be detected as autocorrelated."""
    rng = np.random.default_rng(42)
    n = 200
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = 0.8 * x[i - 1] + rng.standard_normal()
    result = ljung_box(x, n_lags=10)
    assert result.p_value < 0.05


def test_ljung_alias():
    assert ljung is ljung_box
