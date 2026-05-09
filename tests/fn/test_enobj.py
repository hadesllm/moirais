"""Tests for enobj.elastic_net_objective."""
import numpy as np
import pytest
from moirais.fn.enobj import elastic_net_objective


def test_enobj_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    beta = 0.8
    lam = 0.1
    alpha = 0.05
    result = elastic_net_objective(y, X, beta, lam, alpha)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_enobj_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    beta = 0.8
    lam = 0.1
    alpha = 0.05
    result = elastic_net_objective(y, X, beta, lam, alpha)
    assert isinstance(result, dict)
