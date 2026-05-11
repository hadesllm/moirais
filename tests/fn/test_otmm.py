"""Tests for otmm.ot_minibatch_loss."""
import numpy as np
import pytest
from morie.fn.otmm import ot_minibatch_loss


def test_otmm_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    Y = np.random.default_rng(43).normal(0, 1, 100)
    batch_size = 100
    n_batches = np.random.default_rng(42).normal(0, 1, 100)
    epsilon = 1e-6
    result = ot_minibatch_loss(X, Y, batch_size, n_batches, epsilon)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_otmm_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    Y = np.random.default_rng(43).normal(0, 1, 100)
    batch_size = 100
    n_batches = np.random.default_rng(42).normal(0, 1, 100)
    epsilon = 1e-6
    result = ot_minibatch_loss(X, Y, batch_size, n_batches, epsilon)
    assert isinstance(result, dict)
