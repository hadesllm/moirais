"""Tests for gpdD.gpd_distribution."""
import numpy as np
import pytest
from morie.fn.gpdD import gpd_distribution


def test_gpdD_basic():
    """Test basic functionality."""
    sigma = 1.0
    xi = np.random.default_rng(42).normal(0, 1, 100)
    result = gpd_distribution(sigma, xi)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_gpdD_edge():
    """Test edge cases."""
    sigma = 1.0
    xi = np.random.default_rng(42).normal(0, 1, 100)
    result = gpd_distribution(sigma, xi)
    assert isinstance(result, dict)
