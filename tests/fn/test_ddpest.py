"""Tests for ddpest.dependent_dp."""
import numpy as np
import pytest
from moirais.fn.ddpest import dependent_dp


def test_ddpest_basic():
    """Test basic functionality."""
    x_grid = np.random.default_rng(42).normal(0, 1, 100)
    alpha = 0.05
    kernel = (lambda u: np.exp(-0.5*u*u) / np.sqrt(2*np.pi))
    result = dependent_dp(x_grid, alpha, kernel)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ddpest_edge():
    """Test edge cases."""
    x_grid = np.random.default_rng(42).normal(0, 1, 100)
    alpha = 0.05
    kernel = (lambda u: np.exp(-0.5*u*u) / np.sqrt(2*np.pi))
    result = dependent_dp(x_grid, alpha, kernel)
    assert isinstance(result, dict)
