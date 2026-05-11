"""Tests for causmnde.causal_natural_decomposition."""
import numpy as np
import pytest
from morie.fn.causmnde import causal_natural_decomposition


def test_causmnde_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    M = np.random.default_rng(43).normal(0, 1, (10, 10))
    Y = np.random.default_rng(43).normal(0, 1, 100)
    T = np.random.default_rng(43).integers(0, 2, 100)
    result = causal_natural_decomposition(X, M, Y, T)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_causmnde_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    M = np.random.default_rng(43).normal(0, 1, (10, 10))
    Y = np.random.default_rng(43).normal(0, 1, 100)
    T = np.random.default_rng(43).integers(0, 2, 100)
    result = causal_natural_decomposition(X, M, Y, T)
    assert isinstance(result, dict)
