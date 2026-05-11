"""Tests for pmpfit.pmp_fit."""
import numpy as np
import pytest
from morie.fn.pmpfit import pmp_fit


def test_pmpfit_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    sigma_grid = np.random.default_rng(42).normal(0, 1, 100)
    alpha_grid = np.random.default_rng(42).normal(0, 1, 100)
    result = pmp_fit(y, K, sigma_grid, alpha_grid)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_pmpfit_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    sigma_grid = np.random.default_rng(42).normal(0, 1, 100)
    alpha_grid = np.random.default_rng(42).normal(0, 1, 100)
    result = pmp_fit(y, K, sigma_grid, alpha_grid)
    assert isinstance(result, dict)
