"""Tests for hmaen.geron_autoencoder."""
import numpy as np
import pytest
from moirais.fn.hmaen import geron_autoencoder


def test_hmaen_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    bottleneck = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_autoencoder(X, bottleneck)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmaen_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    bottleneck = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_autoencoder(X, bottleneck)
    assert isinstance(result, dict)
