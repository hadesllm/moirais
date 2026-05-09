"""Tests for agclop.alphazero_optimizer."""
import numpy as np
import pytest
from moirais.fn.agclop import alphazero_optimizer


def test_agclop_basic():
    """Test basic functionality."""
    theta = 0.0
    grad = np.random.default_rng(42).normal(0, 1, 100)
    momentum = np.random.default_rng(42).normal(0, 1, 100)
    weight_decay = np.random.default_rng(42).normal(0, 1, 100)
    result = alphazero_optimizer(theta, grad, momentum, weight_decay)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_agclop_edge():
    """Test edge cases."""
    theta = 0.0
    grad = np.random.default_rng(42).normal(0, 1, 100)
    momentum = np.random.default_rng(42).normal(0, 1, 100)
    weight_decay = np.random.default_rng(42).normal(0, 1, 100)
    result = alphazero_optimizer(theta, grad, momentum, weight_decay)
    assert isinstance(result, dict)
