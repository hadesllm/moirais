"""Tests for grael.geron_autoencoder_reconstruction_loss."""
import numpy as np
import pytest
from morie.fn.grael import geron_autoencoder_reconstruction_loss


def test_grael_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    encoded = np.random.default_rng(42).normal(0, 1, 100)
    decoded = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_autoencoder_reconstruction_loss(X, encoded, decoded)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_grael_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    encoded = np.random.default_rng(42).normal(0, 1, 100)
    decoded = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_autoencoder_reconstruction_loss(X, encoded, decoded)
    assert isinstance(result, dict)
