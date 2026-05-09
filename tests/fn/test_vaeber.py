"""Tests for vaeber.vae_elbo."""
import numpy as np
import pytest
from moirais.fn.vaeber import vae_elbo


def test_vaeber_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    encoder = np.random.default_rng(42).normal(0, 1, 100)
    decoder = np.random.default_rng(42).normal(0, 1, 100)
    result = vae_elbo(x, encoder, decoder)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_vaeber_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    encoder = np.random.default_rng(42).normal(0, 1, 100)
    decoder = np.random.default_rng(42).normal(0, 1, 100)
    result = vae_elbo(x, encoder, decoder)
    assert isinstance(result, dict)
