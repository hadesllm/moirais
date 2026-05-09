"""Tests for moirais.fn.pacf — Partial ACF."""
import numpy as np

from moirais.fn.pacf import partial_acf, pacf


def test_pacf_lag0_is_one():
    """PACF at lag 0 should be 1."""
    rng = np.random.default_rng(42)
    x = rng.standard_normal(200)
    result = partial_acf(x, max_lag=10)
    assert abs(result.values[0] - 1.0) < 1e-10


def test_ar1_pacf_cuts_off():
    """AR(1) process: PACF should cut off after lag 1."""
    rng = np.random.default_rng(42)
    n = 1000
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = 0.7 * x[i - 1] + rng.standard_normal()
    result = partial_acf(x, max_lag=5)
    # Lag 1 should be significant, lag 2+ should be near zero
    assert abs(result.values[1]) > 0.3
    assert abs(result.values[3]) < 0.2


def test_pacf_alias():
    assert pacf is partial_acf
