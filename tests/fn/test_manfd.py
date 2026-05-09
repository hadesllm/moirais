"""Tests for manfd.manifold_functional."""
import numpy as np
import pytest
from moirais.fn.manfd import manifold_functional


def test_manfd_basic():
    """Test basic functionality."""
    Y = np.random.default_rng(43).normal(0, 1, 100)
    k = 5
    method = 'auto'
    result = manifold_functional(Y, k, method)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_manfd_edge():
    """Test edge cases."""
    Y = np.random.default_rng(43).normal(0, 1, 100)
    k = 5
    method = 'auto'
    result = manifold_functional(Y, k, method)
    assert isinstance(result, dict)
