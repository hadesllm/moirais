"""Tests for grcae.geron_convolutional_autoencoder."""
import numpy as np
import pytest
from moirais.fn.grcae import geron_convolutional_autoencoder


def test_grcae_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    encoder_weights = np.random.default_rng(42).normal(0, 1, 100)
    decoder_weights = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_convolutional_autoencoder(x, encoder_weights, decoder_weights)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_grcae_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    encoder_weights = np.random.default_rng(42).normal(0, 1, 100)
    decoder_weights = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_convolutional_autoencoder(x, encoder_weights, decoder_weights)
    assert isinstance(result, dict)
