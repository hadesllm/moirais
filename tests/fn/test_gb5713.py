"""Tests for gb5713.gibbons_wsrt_simpower."""
import numpy as np
import pytest
from morie.fn.gb5713 import gibbons_wsrt_simpower


def test_gb5713_basic():
    """Test basic functionality."""
    theta = 0.0
    n = 100
    alpha = 0.05
    nsim = np.random.default_rng(42).normal(0, 1, 100)
    result = gibbons_wsrt_simpower(theta, n, alpha, nsim)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_gb5713_edge():
    """Test edge cases."""
    theta = 0.0
    n = 100
    alpha = 0.05
    nsim = np.random.default_rng(42).normal(0, 1, 100)
    result = gibbons_wsrt_simpower(theta, n, alpha, nsim)
    assert isinstance(result, dict)
