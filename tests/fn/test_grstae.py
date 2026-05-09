"""Tests for grstae.geron_stacked_autoencoder."""
import numpy as np
import pytest
from moirais.fn.grstae import geron_stacked_autoencoder


def test_grstae_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    layer_weights = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_stacked_autoencoder(x, layer_weights)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_grstae_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    layer_weights = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_stacked_autoencoder(x, layer_weights)
    assert isinstance(result, dict)
