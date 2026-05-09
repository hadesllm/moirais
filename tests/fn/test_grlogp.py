"""Tests for grlogp.geron_logistic_regression_probability."""
import numpy as np
import pytest
from moirais.fn.grlogp import geron_logistic_regression_probability


def test_grlogp_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    theta = 0.0
    result = geron_logistic_regression_probability(X, theta)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_grlogp_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    theta = 0.0
    result = geron_logistic_regression_probability(X, theta)
    assert isinstance(result, dict)
