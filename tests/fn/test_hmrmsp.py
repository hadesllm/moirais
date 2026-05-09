"""Tests for hmrmsp.geron_rmsprop."""
import numpy as np
import pytest
from moirais.fn.hmrmsp import geron_rmsprop


def test_hmrmsp_basic():
    """Test basic functionality."""
    grads = np.random.default_rng(42).normal(0, 1, 100)
    s = 90
    beta = 0.8
    eta = np.random.default_rng(42).normal(0, 1, 100)
    eps = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_rmsprop(grads, s, beta, eta, eps)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmrmsp_edge():
    """Test edge cases."""
    grads = np.random.default_rng(42).normal(0, 1, 100)
    s = 90
    beta = 0.8
    eta = np.random.default_rng(42).normal(0, 1, 100)
    eps = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_rmsprop(grads, s, beta, eta, eps)
    assert isinstance(result, dict)
