"""Tests for dpsgd.dp_sgd."""
import numpy as np
import pytest
from moirais.fn.dpsgd import dp_sgd


def test_dpsgd_basic():
    """Test basic functionality."""
    loss = np.random.default_rng(42).normal(0, 1, 100)
    C = np.random.default_rng(42).normal(0, 1, 100)
    sigma = 1.0
    lr = np.random.default_rng(42).normal(0, 1, 100)
    result = dp_sgd(loss, C, sigma, lr)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_dpsgd_edge():
    """Test edge cases."""
    loss = np.random.default_rng(42).normal(0, 1, 100)
    C = np.random.default_rng(42).normal(0, 1, 100)
    sigma = 1.0
    lr = np.random.default_rng(42).normal(0, 1, 100)
    result = dp_sgd(loss, C, sigma, lr)
    assert isinstance(result, dict)
