"""Tests for volsabr.vol_sabr_implied."""
import numpy as np
import pytest
from moirais.fn.volsabr import vol_sabr_implied


def test_volsabr_basic():
    """Test basic functionality."""
    F = np.random.default_rng(43).normal(0, 1, 100)
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    T = np.random.default_rng(43).integers(0, 2, 100)
    alpha = 0.05
    beta = 0.8
    rho = 0.5
    nu = np.random.default_rng(42).normal(0, 1, 100)
    result = vol_sabr_implied(F, K, T, alpha, beta, rho, nu)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_volsabr_edge():
    """Test edge cases."""
    F = np.random.default_rng(43).normal(0, 1, 100)
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    T = np.random.default_rng(43).integers(0, 2, 100)
    alpha = 0.05
    beta = 0.8
    rho = 0.5
    nu = np.random.default_rng(42).normal(0, 1, 100)
    result = vol_sabr_implied(F, K, T, alpha, beta, rho, nu)
    assert isinstance(result, dict)
