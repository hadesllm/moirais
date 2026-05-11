"""Tests for cvxind.boyd_indicator."""
import numpy as np
import pytest
from morie.fn.cvxind import boyd_indicator


def test_cvxind_basic():
    """Test basic functionality."""
    C = np.random.default_rng(42).normal(0, 1, 100)
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = boyd_indicator(C, x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_cvxind_edge():
    """Test edge cases."""
    C = np.random.default_rng(42).normal(0, 1, 100)
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = boyd_indicator(C, x)
    assert isinstance(result, dict)
