"""Tests for grmom.geron_momentum_update."""
import numpy as np
import pytest
from moirais.fn.grmom import geron_momentum_update


def test_grmom_basic():
    """Test basic functionality."""
    theta = 0.0
    grad = np.random.default_rng(42).normal(0, 1, 100)
    v = np.random.default_rng(44).normal(0, 1, 100)
    eta = np.random.default_rng(42).normal(0, 1, 100)
    beta = 0.8
    result = geron_momentum_update(theta, grad, v, eta, beta)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_grmom_edge():
    """Test edge cases."""
    theta = 0.0
    grad = np.random.default_rng(42).normal(0, 1, 100)
    v = np.random.default_rng(44).normal(0, 1, 100)
    eta = np.random.default_rng(42).normal(0, 1, 100)
    beta = 0.8
    result = geron_momentum_update(theta, grad, v, eta, beta)
    assert isinstance(result, dict)
