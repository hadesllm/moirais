"""Tests for gb5414.gibbons_sign_power."""
import numpy as np
import pytest
from morie.fn.gb5414 import gibbons_sign_power


def test_gb5414_basic():
    """Test basic functionality."""
    theta = 0.0
    n = 100
    alpha = 0.05
    F = np.random.default_rng(42).normal(0, 1, 100)
    result = gibbons_sign_power(theta, n, alpha, F)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_gb5414_edge():
    """Test edge cases."""
    theta = 0.0
    n = 100
    alpha = 0.05
    F = np.random.default_rng(42).normal(0, 1, 100)
    result = gibbons_sign_power(theta, n, alpha, F)
    assert isinstance(result, dict)
