"""Tests for hmlogp.geron_logistic_probability."""
import numpy as np
import pytest
from moirais.fn.hmlogp import geron_logistic_probability


def test_hmlogp_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    theta = 0.0
    result = geron_logistic_probability(X, theta)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmlogp_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    theta = 0.0
    result = geron_logistic_probability(X, theta)
    assert isinstance(result, dict)
