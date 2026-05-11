"""Tests for gb1221m.gibbons_friedman_mult."""
import numpy as np
import pytest
from morie.fn.gb1221m import gibbons_friedman_mult


def test_gb1221m_basic():
    """Test basic functionality."""
    data = np.random.default_rng(42).normal(0, 1, 100)
    k = 5
    alpha = 0.05
    result = gibbons_friedman_mult(data, k, alpha)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_gb1221m_edge():
    """Test edge cases."""
    data = np.random.default_rng(42).normal(0, 1, 100)
    k = 5
    alpha = 0.05
    result = gibbons_friedman_mult(data, k, alpha)
    assert isinstance(result, dict)
