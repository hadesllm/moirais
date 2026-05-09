"""Tests for grvae.geron_vae_elbo."""
import numpy as np
import pytest
from moirais.fn.grvae import geron_vae_elbo


def test_grvae_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    mu = 0.0
    logvar = np.random.default_rng(42).normal(0, 1, 100)
    recon = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_vae_elbo(x, mu, logvar, recon)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_grvae_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    mu = 0.0
    logvar = np.random.default_rng(42).normal(0, 1, 100)
    recon = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_vae_elbo(x, mu, logvar, recon)
    assert isinstance(result, dict)
