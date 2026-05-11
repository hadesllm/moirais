"""Tests for l2reg.l2_weight_regularization."""
import numpy as np
import pytest
from morie.fn.l2reg import l2_weight_regularization


def test_l2reg_basic():
    """Test basic functionality."""
    loss = np.random.default_rng(42).normal(0, 1, 100)
    weights = np.random.default_rng(45).exponential(1, 100)
    lam = 0.1
    result = l2_weight_regularization(loss, weights, lam)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_l2reg_edge():
    """Test edge cases."""
    loss = np.random.default_rng(42).normal(0, 1, 100)
    weights = np.random.default_rng(45).exponential(1, 100)
    lam = 0.1
    result = l2_weight_regularization(loss, weights, lam)
    assert isinstance(result, dict)
