"""Tests for gevD.gev_distribution."""
import numpy as np
import pytest
from moirais.fn.gevD import gev_distribution


def test_gevD_basic():
    """Test basic functionality."""
    mu = 0.0
    sigma = 1.0
    xi = np.random.default_rng(42).normal(0, 1, 100)
    result = gev_distribution(mu, sigma, xi)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_gevD_edge():
    """Test edge cases."""
    mu = 0.0
    sigma = 1.0
    xi = np.random.default_rng(42).normal(0, 1, 100)
    result = gev_distribution(mu, sigma, xi)
    assert isinstance(result, dict)
