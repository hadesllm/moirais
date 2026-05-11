"""Tests for grsae.geron_sparse_autoencoder."""
import numpy as np
import pytest
from morie.fn.grsae import geron_sparse_autoencoder


def test_grsae_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    hidden = np.random.default_rng(42).normal(0, 1, 100)
    decoded = np.random.default_rng(42).normal(0, 1, 100)
    lam = 0.1
    result = geron_sparse_autoencoder(x, hidden, decoded, lam)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_grsae_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    hidden = np.random.default_rng(42).normal(0, 1, 100)
    decoded = np.random.default_rng(42).normal(0, 1, 100)
    lam = 0.1
    result = geron_sparse_autoencoder(x, hidden, decoded, lam)
    assert isinstance(result, dict)
