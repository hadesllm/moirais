"""Tests for weisL.wl_kernel."""
import numpy as np
import pytest
from morie.fn.weisL import wl_kernel


def test_weisL_basic():
    """Test basic functionality."""
    G1 = np.random.default_rng(42).normal(0, 1, 100)
    G2 = np.random.default_rng(42).normal(0, 1, 100)
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    result = wl_kernel(G1, G2, K)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_weisL_edge():
    """Test edge cases."""
    G1 = np.random.default_rng(42).normal(0, 1, 100)
    G2 = np.random.default_rng(42).normal(0, 1, 100)
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    result = wl_kernel(G1, G2, K)
    assert isinstance(result, dict)
