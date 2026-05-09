"""Tests for gb531.gibbons_quantile_test."""
import numpy as np
import pytest
from moirais.fn.gb531 import gibbons_quantile_test


def test_gb531_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    p = 5
    x_p0 = np.random.default_rng(42).normal(0, 1, 100)
    alpha = 0.05
    result = gibbons_quantile_test(x, p, x_p0, alpha)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_gb531_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    p = 5
    x_p0 = np.random.default_rng(42).normal(0, 1, 100)
    alpha = 0.05
    result = gibbons_quantile_test(x, p, x_p0, alpha)
    assert isinstance(result, dict)
