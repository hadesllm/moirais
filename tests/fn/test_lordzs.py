"""Tests for lordzs.lord_chi_square."""
import numpy as np
import pytest
from moirais.fn.lordzs import lord_chi_square


def test_lordzs_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    b_R = np.random.default_rng(42).normal(0, 1, 100)
    b_F = np.random.default_rng(42).normal(0, 1, 100)
    V = np.random.default_rng(42).normal(0, 1, 100)
    result = lord_chi_square(y, b_R, b_F, V)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_lordzs_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    b_R = np.random.default_rng(42).normal(0, 1, 100)
    b_F = np.random.default_rng(42).normal(0, 1, 100)
    V = np.random.default_rng(42).normal(0, 1, 100)
    result = lord_chi_square(y, b_R, b_F, V)
    assert isinstance(result, dict)
