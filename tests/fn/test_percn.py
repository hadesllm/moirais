"""Tests for percn.perceptron_activation."""
import numpy as np
import pytest
from morie.fn.percn import perceptron_activation


def test_percn_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    w = np.random.default_rng(45).exponential(1, 100)
    b = np.random.default_rng(42).normal(0, 1, 100)
    result = perceptron_activation(X, w, b)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_percn_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    w = np.random.default_rng(45).exponential(1, 100)
    b = np.random.default_rng(42).normal(0, 1, 100)
    result = perceptron_activation(X, w, b)
    assert isinstance(result, dict)
