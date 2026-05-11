"""Tests for sgdup.sgd_update."""
import numpy as np
import pytest
from morie.fn.sgdup import sgd_update


def test_sgdup_basic():
    """Test basic functionality."""
    beta = 0.8
    batch_grads = np.random.default_rng(42).normal(0, 1, 100)
    eta = np.random.default_rng(42).normal(0, 1, 100)
    result = sgd_update(beta, batch_grads, eta)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_sgdup_edge():
    """Test edge cases."""
    beta = 0.8
    batch_grads = np.random.default_rng(42).normal(0, 1, 100)
    eta = np.random.default_rng(42).normal(0, 1, 100)
    result = sgd_update(beta, batch_grads, eta)
    assert isinstance(result, dict)
