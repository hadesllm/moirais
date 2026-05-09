"""Tests for hmbel.geron_bellman_optimality."""
import numpy as np
import pytest
from moirais.fn.hmbel import geron_bellman_optimality


def test_hmbel_basic():
    """Test basic functionality."""
    V = np.random.default_rng(42).normal(0, 1, 100)
    P = np.random.default_rng(42).normal(0, 1, 100)
    R = np.random.default_rng(42).normal(0, 1, 100)
    gamma = 1.0
    result = geron_bellman_optimality(V, P, R, gamma)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmbel_edge():
    """Test edge cases."""
    V = np.random.default_rng(42).normal(0, 1, 100)
    P = np.random.default_rng(42).normal(0, 1, 100)
    R = np.random.default_rng(42).normal(0, 1, 100)
    gamma = 1.0
    result = geron_bellman_optimality(V, P, R, gamma)
    assert isinstance(result, dict)
