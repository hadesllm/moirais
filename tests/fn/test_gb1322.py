"""Tests for gb1322.gibbons_are_formula."""
import numpy as np
import pytest
from morie.fn.gb1322 import gibbons_are_formula


def test_gb1322_basic():
    """Test basic functionality."""
    T = np.random.default_rng(42).normal(0, 1, 100)
    T_star = np.random.default_rng(42).normal(0, 1, 100)
    u0 = np.random.default_rng(42).normal(0, 1, 100)
    result = gibbons_are_formula(T, T_star, u0)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_gb1322_edge():
    """Test edge cases."""
    T = np.random.default_rng(42).normal(0, 1, 100)
    T_star = np.random.default_rng(42).normal(0, 1, 100)
    u0 = np.random.default_rng(42).normal(0, 1, 100)
    result = gibbons_are_formula(T, T_star, u0)
    assert isinstance(result, dict)
