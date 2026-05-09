"""Tests for hmdae.geron_denoising_autoencoder."""
import numpy as np
import pytest
from moirais.fn.hmdae import geron_denoising_autoencoder


def test_hmdae_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    noise_std = np.random.default_rng(42).normal(0, 1, 100)
    epochs = np.random.default_rng(42).normal(0, 1, 100)
    lr = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_denoising_autoencoder(X, noise_std, epochs, lr)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmdae_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    noise_std = np.random.default_rng(42).normal(0, 1, 100)
    epochs = np.random.default_rng(42).normal(0, 1, 100)
    lr = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_denoising_autoencoder(X, noise_std, epochs, lr)
    assert isinstance(result, dict)
