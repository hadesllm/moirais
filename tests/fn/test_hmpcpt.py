"""Tests for hmpcpt.geron_perceptron."""
import numpy as np
import pytest
from moirais.fn.hmpcpt import geron_perceptron


def test_hmpcpt_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    eta = np.random.default_rng(42).normal(0, 1, 100)
    n_iter = 50
    result = geron_perceptron(X, y, eta, n_iter)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmpcpt_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    eta = np.random.default_rng(42).normal(0, 1, 100)
    n_iter = 50
    result = geron_perceptron(X, y, eta, n_iter)
    assert isinstance(result, dict)
