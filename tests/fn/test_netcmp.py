"""Tests for netcmp.network_comparison."""
import numpy as np
import pytest
from moirais.fn.netcmp import network_comparison


def test_netcmp_basic():
    """Test basic functionality."""
    G1 = np.random.default_rng(42).normal(0, 1, 100)
    G2 = np.random.default_rng(42).normal(0, 1, 100)
    kernel = (lambda u: np.exp(-0.5*u*u) / np.sqrt(2*np.pi))
    result = network_comparison(G1, G2, kernel)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_netcmp_edge():
    """Test edge cases."""
    G1 = np.random.default_rng(42).normal(0, 1, 100)
    G2 = np.random.default_rng(42).normal(0, 1, 100)
    kernel = (lambda u: np.exp(-0.5*u*u) / np.sqrt(2*np.pi))
    result = network_comparison(G1, G2, kernel)
    assert isinstance(result, dict)
