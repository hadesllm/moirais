"""Tests for hmspae.geron_sparse_autoencoder."""
import numpy as np
import pytest
from moirais.fn.hmspae import geron_sparse_autoencoder


def test_hmspae_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    rho = 0.5
    beta = 0.8
    epochs = np.random.default_rng(42).normal(0, 1, 100)
    lr = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_sparse_autoencoder(X, rho, beta, epochs, lr)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmspae_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    rho = 0.5
    beta = 0.8
    epochs = np.random.default_rng(42).normal(0, 1, 100)
    lr = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_sparse_autoencoder(X, rho, beta, epochs, lr)
    assert isinstance(result, dict)
