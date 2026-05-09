"""Tests for betvae.beta_vae_disentangle."""
import numpy as np
import pytest
from moirais.fn.betvae import beta_vae_disentangle


def test_betvae_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    encoder = np.random.default_rng(42).normal(0, 1, 100)
    decoder = np.random.default_rng(42).normal(0, 1, 100)
    beta = 0.8
    result = beta_vae_disentangle(x, encoder, decoder, beta)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_betvae_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    encoder = np.random.default_rng(42).normal(0, 1, 100)
    decoder = np.random.default_rng(42).normal(0, 1, 100)
    beta = 0.8
    result = beta_vae_disentangle(x, encoder, decoder, beta)
    assert isinstance(result, dict)
