"""Tests for grn013.geron_ch4_elastic_net_cost_function."""
import numpy as np
import pytest
from morie.fn.grn013 import geron_ch4_elastic_net_cost_function


def test_grn013_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    theta = 0.0
    alpha = 0.05
    r = 10
    result = geron_ch4_elastic_net_cost_function(X, y, theta, alpha, r)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_grn013_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    theta = 0.0
    alpha = 0.05
    r = 10
    result = geron_ch4_elastic_net_cost_function(X, y, theta, alpha, r)
    assert isinstance(result, dict)
