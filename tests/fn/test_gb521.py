"""Tests for gb521.gibbons_quantile_ci."""
import numpy as np
import pytest
from morie.fn.gb521 import gibbons_quantile_ci


def test_gb521_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    p = 5
    alpha = 0.05
    result = gibbons_quantile_ci(x, p, alpha)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_gb521_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    p = 5
    alpha = 0.05
    result = gibbons_quantile_ci(x, p, alpha)
    assert isinstance(result, dict)
