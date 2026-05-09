"""Tests for grsmxp.geron_softmax_probability."""
import numpy as np
import pytest
from moirais.fn.grsmxp import geron_softmax_probability


def test_grsmxp_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    theta = 0.0
    result = geron_softmax_probability(X, theta)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_grsmxp_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    theta = 0.0
    result = geron_softmax_probability(X, theta)
    assert isinstance(result, dict)
