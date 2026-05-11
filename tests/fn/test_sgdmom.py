"""Tests for sgdmom.sgd_momentum."""
import numpy as np
import pytest
from morie.fn.sgdmom import sgd_momentum


def test_sgdmom_basic():
    """Test basic functionality."""
    g = np.random.default_rng(43).normal(0, 1, 100)
    mu = 0.0
    lr = np.random.default_rng(42).normal(0, 1, 100)
    result = sgd_momentum(g, mu, lr)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_sgdmom_edge():
    """Test edge cases."""
    g = np.random.default_rng(43).normal(0, 1, 100)
    mu = 0.0
    lr = np.random.default_rng(42).normal(0, 1, 100)
    result = sgd_momentum(g, mu, lr)
    assert isinstance(result, dict)
