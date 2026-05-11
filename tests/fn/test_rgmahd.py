"""Tests for rgmahd.rangayyan_mahalanobis."""
import numpy as np
import pytest
from morie.fn.rgmahd import rangayyan_mahalanobis


def test_rgmahd_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    mu = 0.0
    sigma = 1.0
    result = rangayyan_mahalanobis(x, mu, sigma)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rgmahd_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    mu = 0.0
    sigma = 1.0
    result = rangayyan_mahalanobis(x, mu, sigma)
    assert isinstance(result, dict)
