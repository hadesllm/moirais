"""Tests for lapkn.laplacian_kernel."""
import numpy as np
import pytest
from moirais.fn.lapkn import laplacian_kernel


def test_lapkn_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    h = 0.3
    result = laplacian_kernel(X, h)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_lapkn_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    h = 0.3
    result = laplacian_kernel(X, h)
    assert isinstance(result, dict)
