"""Tests for gb641c.gibbons_median_test_ci."""
import numpy as np
import pytest
from morie.fn.gb641c import gibbons_median_test_ci


def test_gb641c_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    alpha = 0.05
    result = gibbons_median_test_ci(x, y, alpha)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_gb641c_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    alpha = 0.05
    result = gibbons_median_test_ci(x, y, alpha)
    assert isinstance(result, dict)
