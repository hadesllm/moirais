"""Tests for grelas.geron_elastic_net_cost."""
import numpy as np
import pytest
from moirais.fn.grelas import geron_elastic_net_cost


def test_grelas_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    theta = 0.0
    alpha = 0.05
    r = 10
    result = geron_elastic_net_cost(X, y, theta, alpha, r)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_grelas_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    theta = 0.0
    alpha = 0.05
    r = 10
    result = geron_elastic_net_cost(X, y, theta, alpha, r)
    assert isinstance(result, dict)
