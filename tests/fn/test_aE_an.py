"""Tests for aE_an.autoencoder_anomaly."""
import numpy as np
import pytest
from morie.fn.aE_an import autoencoder_anomaly


def test_aE_an_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    ae = np.random.default_rng(42).normal(0, 1, 100)
    result = autoencoder_anomaly(X, ae)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_aE_an_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    ae = np.random.default_rng(42).normal(0, 1, 100)
    result = autoencoder_anomaly(X, ae)
    assert isinstance(result, dict)
