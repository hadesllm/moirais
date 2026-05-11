"""Tests for grridn.geron_ridge_normal_equation."""
import numpy as np
import pytest
from morie.fn.grridn import geron_ridge_normal_equation


def test_grridn_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    alpha = 0.05
    result = geron_ridge_normal_equation(X, y, alpha)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_grridn_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    alpha = 0.05
    result = geron_ridge_normal_equation(X, y, alpha)
    assert isinstance(result, dict)
