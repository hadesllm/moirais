"""Tests for moirais.fn.durbin — Durbin-Watson statistic."""
import numpy as np
import pytest

from moirais.fn.durbin import durbin_watson, durbin


def test_random_residuals_near_two():
    """White noise residuals should give DW near 2."""
    rng = np.random.default_rng(42)
    e = rng.standard_normal(200)
    result = durbin_watson(e)
    assert 1.5 < result.value < 2.5
    assert result.extra["interpretation"] == "no autocorrelation"


def test_ar1_residuals_below_two():
    """AR(1) residuals with positive autocorrelation should give DW < 2."""
    rng = np.random.default_rng(42)
    n = 200
    e = np.zeros(n)
    for i in range(1, n):
        e[i] = 0.8 * e[i - 1] + rng.standard_normal()
    result = durbin_watson(e)
    assert result.value < 1.5


def test_durbin_alias():
    assert durbin is durbin_watson


def test_too_few_residuals():
    with pytest.raises(ValueError):
        durbin_watson([1, 2])
