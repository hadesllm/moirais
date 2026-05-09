"""Tests for lambopt.lamb_optimizer."""
import numpy as np
import pytest
from moirais.fn.lambopt import lamb_optimizer


def test_lambopt_basic():
    """Test basic functionality."""
    g = np.random.default_rng(43).normal(0, 1, 100)
    layer_idx = np.random.default_rng(42).normal(0, 1, 100)
    lr = np.random.default_rng(42).normal(0, 1, 100)
    result = lamb_optimizer(g, layer_idx, lr)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_lambopt_edge():
    """Test edge cases."""
    g = np.random.default_rng(43).normal(0, 1, 100)
    layer_idx = np.random.default_rng(42).normal(0, 1, 100)
    lr = np.random.default_rng(42).normal(0, 1, 100)
    result = lamb_optimizer(g, layer_idx, lr)
    assert isinstance(result, dict)
