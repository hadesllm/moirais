"""Tests for hmmnl.geron_multinomial_logistic."""
import numpy as np
import pytest
from moirais.fn.hmmnl import geron_multinomial_logistic


def test_hmmnl_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    Y = np.random.default_rng(43).normal(0, 1, 100)
    lr = np.random.default_rng(42).normal(0, 1, 100)
    n_iter = 50
    result = geron_multinomial_logistic(X, Y, lr, n_iter)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmmnl_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    Y = np.random.default_rng(43).normal(0, 1, 100)
    lr = np.random.default_rng(42).normal(0, 1, 100)
    n_iter = 50
    result = geron_multinomial_logistic(X, Y, lr, n_iter)
    assert isinstance(result, dict)
