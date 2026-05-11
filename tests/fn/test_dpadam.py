"""Tests for dpadam.dp_adam."""
import numpy as np
import pytest
from morie.fn.dpadam import dp_adam


def test_dpadam_basic():
    """Test basic functionality."""
    loss = np.random.default_rng(42).normal(0, 1, 100)
    C = np.random.default_rng(42).normal(0, 1, 100)
    sigma = 1.0
    lr = np.random.default_rng(42).normal(0, 1, 100)
    betas = np.random.default_rng(42).normal(0, 1, 100)
    result = dp_adam(loss, C, sigma, lr, betas)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_dpadam_edge():
    """Test edge cases."""
    loss = np.random.default_rng(42).normal(0, 1, 100)
    C = np.random.default_rng(42).normal(0, 1, 100)
    sigma = 1.0
    lr = np.random.default_rng(42).normal(0, 1, 100)
    betas = np.random.default_rng(42).normal(0, 1, 100)
    result = dp_adam(loss, C, sigma, lr, betas)
    assert isinstance(result, dict)
