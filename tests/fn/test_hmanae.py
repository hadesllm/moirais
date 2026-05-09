"""Tests for hmanae.geron_anomaly_autoencoder."""
import numpy as np
import pytest
from moirais.fn.hmanae import geron_anomaly_autoencoder


def test_hmanae_basic():
    """Test basic functionality."""
    model = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    threshold = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_anomaly_autoencoder(model, X, threshold)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmanae_edge():
    """Test edge cases."""
    model = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    threshold = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_anomaly_autoencoder(model, X, threshold)
    assert isinstance(result, dict)
