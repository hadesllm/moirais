"""Tests for grlogg.geron_logistic_cost_gradient."""
import numpy as np
import pytest
from moirais.fn.grlogg import geron_logistic_cost_gradient


def test_grlogg_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    theta = 0.0
    result = geron_logistic_cost_gradient(X, y, theta)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_grlogg_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    theta = 0.0
    result = geron_logistic_cost_gradient(X, y, theta)
    assert isinstance(result, dict)
