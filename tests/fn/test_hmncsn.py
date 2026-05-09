"""Tests for hmncsn.geron_ncsn."""
import numpy as np
import pytest
from moirais.fn.hmncsn import geron_ncsn


def test_hmncsn_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    sigmas = np.random.default_rng(42).normal(0, 1, 100)
    epochs = np.random.default_rng(42).normal(0, 1, 100)
    lr = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_ncsn(X, sigmas, epochs, lr)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmncsn_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    sigmas = np.random.default_rng(42).normal(0, 1, 100)
    epochs = np.random.default_rng(42).normal(0, 1, 100)
    lr = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_ncsn(X, sigmas, epochs, lr)
    assert isinstance(result, dict)
