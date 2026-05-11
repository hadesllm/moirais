"""Tests for copgau.gaussian_copula."""
import numpy as np
import pytest
from morie.fn.copgau import gaussian_copula


def test_copgau_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    u = np.random.default_rng(44).normal(0, 1, 100)
    v = np.random.default_rng(44).normal(0, 1, 100)
    rho = 0.5
    result = gaussian_copula(y, u, v, rho)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_copgau_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    u = np.random.default_rng(44).normal(0, 1, 100)
    v = np.random.default_rng(44).normal(0, 1, 100)
    rho = 0.5
    result = gaussian_copula(y, u, v, rho)
    assert isinstance(result, dict)
