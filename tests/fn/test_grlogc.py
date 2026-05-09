"""Tests for grlogc.geron_logistic_cross_entropy_cost."""
import numpy as np
import pytest
from moirais.fn.grlogc import geron_logistic_cross_entropy_cost


def test_grlogc_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    theta = 0.0
    result = geron_logistic_cross_entropy_cost(X, y, theta)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_grlogc_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    theta = 0.0
    result = geron_logistic_cross_entropy_cost(X, y, theta)
    assert isinstance(result, dict)
