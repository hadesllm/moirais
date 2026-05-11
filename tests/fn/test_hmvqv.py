"""Tests for hmvqv.geron_vq_vae."""
import numpy as np
import pytest
from morie.fn.hmvqv import geron_vq_vae


def test_hmvqv_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    codebook_size = 100
    latent_dim = 2
    epochs = np.random.default_rng(42).normal(0, 1, 100)
    lr = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_vq_vae(X, codebook_size, latent_dim, epochs, lr)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmvqv_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    codebook_size = 100
    latent_dim = 2
    epochs = np.random.default_rng(42).normal(0, 1, 100)
    lr = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_vq_vae(X, codebook_size, latent_dim, epochs, lr)
    assert isinstance(result, dict)
