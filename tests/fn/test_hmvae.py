"""Tests for hmvae.geron_vae."""
import numpy as np
import pytest
from morie.fn.hmvae import geron_vae


def test_hmvae_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    latent_dim = 2
    epochs = np.random.default_rng(42).normal(0, 1, 100)
    lr = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_vae(X, latent_dim, epochs, lr)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmvae_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    latent_dim = 2
    epochs = np.random.default_rng(42).normal(0, 1, 100)
    lr = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_vae(X, latent_dim, epochs, lr)
    assert isinstance(result, dict)
