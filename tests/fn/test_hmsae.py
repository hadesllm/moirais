"""Tests for hmsae.geron_stacked_autoencoder."""
import numpy as np
import pytest
from moirais.fn.hmsae import geron_stacked_autoencoder


def test_hmsae_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    hidden_sizes = np.random.default_rng(42).normal(0, 1, 100)
    epochs = np.random.default_rng(42).normal(0, 1, 100)
    lr = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_stacked_autoencoder(X, hidden_sizes, epochs, lr)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmsae_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    hidden_sizes = np.random.default_rng(42).normal(0, 1, 100)
    epochs = np.random.default_rng(42).normal(0, 1, 100)
    lr = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_stacked_autoencoder(X, hidden_sizes, epochs, lr)
    assert isinstance(result, dict)
