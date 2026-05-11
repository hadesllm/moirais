"""Tests for sensIM.imai_sensitivity_rho."""
import numpy as np
import pytest
from morie.fn.sensIM import imai_sensitivity_rho


def test_sensIM_basic():
    """Test basic functionality."""
    Y = np.random.default_rng(43).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    M = np.random.default_rng(43).normal(0, 1, (10, 10))
    rho_grid = np.random.default_rng(42).normal(0, 1, 100)
    result = imai_sensitivity_rho(Y, X, M, rho_grid)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_sensIM_edge():
    """Test edge cases."""
    Y = np.random.default_rng(43).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    M = np.random.default_rng(43).normal(0, 1, (10, 10))
    rho_grid = np.random.default_rng(42).normal(0, 1, 100)
    result = imai_sensitivity_rho(Y, X, M, rho_grid)
    assert isinstance(result, dict)
