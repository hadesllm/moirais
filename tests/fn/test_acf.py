"""Tests for morie.fn.acf — Autocorrelation function."""
import numpy as np
import pytest

from morie.fn.acf import autocorrelation, acf


def test_acf_lag0_is_one():
    """ACF at lag 0 should always be 1."""
    rng = np.random.default_rng(42)
    x = rng.standard_normal(100)
    result = autocorrelation(x, max_lag=10)
    assert abs(result.values[0] - 1.0) < 1e-10


def test_ar1_lag1_matches():
    """AR(1) with phi=0.7 should have ACF(1) near 0.7."""
    rng = np.random.default_rng(42)
    n = 1000
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = 0.7 * x[i - 1] + rng.standard_normal()
    result = autocorrelation(x, max_lag=5)
    assert abs(result.values[1] - 0.7) < 0.1


def test_acf_alias():
    assert acf is autocorrelation


def test_too_few_obs():
    with pytest.raises(ValueError):
        autocorrelation([1, 2])
