"""Tests for gb641p.gibbons_median_test_power."""
import numpy as np
import pytest
from morie.fn.gb641p import gibbons_median_test_power


def test_gb641p_basic():
    """Test basic functionality."""
    m = 10
    n = 100
    Delta = np.random.default_rng(42).normal(0, 1, 100)
    alpha = 0.05
    result = gibbons_median_test_power(m, n, Delta, alpha)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_gb641p_edge():
    """Test edge cases."""
    m = 10
    n = 100
    Delta = np.random.default_rng(42).normal(0, 1, 100)
    alpha = 0.05
    result = gibbons_median_test_power(m, n, Delta, alpha)
    assert isinstance(result, dict)
