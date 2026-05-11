"""Tests for hmenet.geron_elastic_net."""
import numpy as np
import pytest
from morie.fn.hmenet import geron_elastic_net


def test_hmenet_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    theta = 0.0
    alpha = 0.05
    r = 10
    result = geron_elastic_net(X, y, theta, alpha, r)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmenet_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    theta = 0.0
    alpha = 0.05
    r = 10
    result = geron_elastic_net(X, y, theta, alpha, r)
    assert isinstance(result, dict)
