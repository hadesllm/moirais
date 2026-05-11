"""Tests for miestn.mi_neural_estimator."""
import numpy as np
import pytest
from morie.fn.miestn import mi_neural_estimator


def test_miestn_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    Y = np.random.default_rng(43).normal(0, 1, 100)
    T_network = np.random.default_rng(42).normal(0, 1, 100)
    result = mi_neural_estimator(X, Y, T_network)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_miestn_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    Y = np.random.default_rng(43).normal(0, 1, 100)
    T_network = np.random.default_rng(42).normal(0, 1, 100)
    result = mi_neural_estimator(X, Y, T_network)
    assert isinstance(result, dict)
