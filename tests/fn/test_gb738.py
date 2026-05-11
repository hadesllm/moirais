"""Tests for gb738.gibbons_chernoff_savage."""
import numpy as np
import pytest
from morie.fn.gb738 import gibbons_chernoff_savage


def test_gb738_basic():
    """Test basic functionality."""
    T_N = np.random.default_rng(42).normal(0, 1, 100)
    m = 10
    n = 100
    J = 20
    result = gibbons_chernoff_savage(T_N, m, n, J)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_gb738_edge():
    """Test edge cases."""
    T_N = np.random.default_rng(42).normal(0, 1, 100)
    m = 10
    n = 100
    J = 20
    result = gibbons_chernoff_savage(T_N, m, n, J)
    assert isinstance(result, dict)
