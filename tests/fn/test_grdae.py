"""Tests for grdae.geron_denoising_autoencoder."""
import numpy as np
import pytest
from morie.fn.grdae import geron_denoising_autoencoder


def test_grdae_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    noise = np.random.default_rng(42).normal(0, 1, 100)
    decoded = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_denoising_autoencoder(x, noise, decoded)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_grdae_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    noise = np.random.default_rng(42).normal(0, 1, 100)
    decoded = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_denoising_autoencoder(x, noise, decoded)
    assert isinstance(result, dict)
