"""Tests for hmcec.geron_cross_entropy_cost."""
import numpy as np
import pytest
from moirais.fn.hmcec import geron_cross_entropy_cost


def test_hmcec_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    Y = np.random.default_rng(43).normal(0, 1, 100)
    theta = 0.0
    result = geron_cross_entropy_cost(X, Y, theta)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmcec_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    Y = np.random.default_rng(43).normal(0, 1, 100)
    theta = 0.0
    result = geron_cross_entropy_cost(X, Y, theta)
    assert isinstance(result, dict)
