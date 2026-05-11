"""Tests for rgrbf.rangayyan_rbf_network."""
import numpy as np
import pytest
from morie.fn.rgrbf import rangayyan_rbf_network


def test_rgrbf_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    n_centers = np.random.default_rng(42).normal(0, 1, 100)
    sigma = 1.0
    result = rangayyan_rbf_network(X, y, n_centers, sigma)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rgrbf_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    n_centers = np.random.default_rng(42).normal(0, 1, 100)
    sigma = 1.0
    result = rangayyan_rbf_network(X, y, n_centers, sigma)
    assert isinstance(result, dict)
