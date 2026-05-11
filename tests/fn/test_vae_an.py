"""Tests for vae_an.vae_anomaly."""
import numpy as np
import pytest
from morie.fn.vae_an import vae_anomaly


def test_vae_an_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    vae = np.random.default_rng(42).normal(0, 1, 100)
    result = vae_anomaly(X, vae)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_vae_an_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    vae = np.random.default_rng(42).normal(0, 1, 100)
    result = vae_anomaly(X, vae)
    assert isinstance(result, dict)
