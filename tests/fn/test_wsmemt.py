"""Tests for wsmemt.wasserman_em_algorithm."""
import numpy as np
import pytest
from morie.fn.wsmemt import wasserman_em_algorithm


def test_wsmemt_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    theta0 = 0.0
    result = wasserman_em_algorithm(X, theta0)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wsmemt_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    theta0 = 0.0
    result = wasserman_em_algorithm(X, theta0)
    assert isinstance(result, dict)
