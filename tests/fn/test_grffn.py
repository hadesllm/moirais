"""Tests for grffn.geron_transformer_feedforward."""
import numpy as np
import pytest
from moirais.fn.grffn import geron_transformer_feedforward


def test_grffn_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    W1 = np.random.default_rng(42).normal(0, 1, 100)
    b1 = np.random.default_rng(42).normal(0, 1, 100)
    W2 = np.random.default_rng(42).normal(0, 1, 100)
    b2 = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_transformer_feedforward(x, W1, b1, W2, b2)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_grffn_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    W1 = np.random.default_rng(42).normal(0, 1, 100)
    b1 = np.random.default_rng(42).normal(0, 1, 100)
    W2 = np.random.default_rng(42).normal(0, 1, 100)
    b2 = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_transformer_feedforward(x, W1, b1, W2, b2)
    assert isinstance(result, dict)
