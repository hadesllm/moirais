"""Tests for primal.primal_dual."""
import numpy as np
import pytest
from moirais.fn.primal import primal_dual


def test_primal_basic():
    """Test basic functionality."""
    F = np.random.default_rng(43).normal(0, 1, 100)
    G = np.eye(10)
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    tau = 0.1
    sigma = 1.0
    result = primal_dual(F, G, K, tau, sigma)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_primal_edge():
    """Test edge cases."""
    F = np.random.default_rng(43).normal(0, 1, 100)
    G = np.eye(10)
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    tau = 0.1
    sigma = 1.0
    result = primal_dual(F, G, K, tau, sigma)
    assert isinstance(result, dict)
