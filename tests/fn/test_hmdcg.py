"""Tests for hmdcg.geron_dcgan."""
import numpy as np
import pytest
from morie.fn.hmdcg import geron_dcgan


def test_hmdcg_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    z_dim = 2
    filters = np.random.default_rng(42).normal(0, 1, 100)
    epochs = np.random.default_rng(42).normal(0, 1, 100)
    lr = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_dcgan(X, z_dim, filters, epochs, lr)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmdcg_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    z_dim = 2
    filters = np.random.default_rng(42).normal(0, 1, 100)
    epochs = np.random.default_rng(42).normal(0, 1, 100)
    lr = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_dcgan(X, z_dim, filters, epochs, lr)
    assert isinstance(result, dict)
