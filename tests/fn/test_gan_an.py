"""Tests for gan_an.gan_anomaly."""
import numpy as np
import pytest
from moirais.fn.gan_an import gan_anomaly


def test_gan_an_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    gan = np.random.default_rng(42).normal(0, 1, 100)
    result = gan_anomaly(X, gan)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_gan_an_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    gan = np.random.default_rng(42).normal(0, 1, 100)
    result = gan_anomaly(X, gan)
    assert isinstance(result, dict)
