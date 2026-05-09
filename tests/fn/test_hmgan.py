"""Tests for hmgan.geron_gan."""
import numpy as np
import pytest
from moirais.fn.hmgan import geron_gan


def test_hmgan_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    G = np.eye(10)
    D = np.random.default_rng(42).normal(0, 1, 100)
    z_dim = 2
    epochs = np.random.default_rng(42).normal(0, 1, 100)
    lr = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_gan(X, G, D, z_dim, epochs, lr)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmgan_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    G = np.eye(10)
    D = np.random.default_rng(42).normal(0, 1, 100)
    z_dim = 2
    epochs = np.random.default_rng(42).normal(0, 1, 100)
    lr = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_gan(X, G, D, z_dim, epochs, lr)
    assert isinstance(result, dict)
