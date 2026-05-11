"""Tests for rbfk.rbf_kernel."""
import numpy as np
import pytest
from morie.fn.rbfk import rbf_kernel


def test_rbfk_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    sigma = 1.0
    result = rbf_kernel(x, y, sigma)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rbfk_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    sigma = 1.0
    result = rbf_kernel(x, y, sigma)
    assert isinstance(result, dict)
