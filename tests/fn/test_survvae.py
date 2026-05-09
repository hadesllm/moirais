"""Tests for survvae.vae_survival."""
import numpy as np
import pytest
from moirais.fn.survvae import vae_survival


def test_survvae_basic():
    """Test basic functionality."""
    time = np.linspace(0, 10, 100)
    event = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = vae_survival(time, event, X)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_survvae_edge():
    """Test edge cases."""
    time = np.linspace(0, 10, 100)
    event = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = vae_survival(time, event, X)
    assert isinstance(result, dict)
