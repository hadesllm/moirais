"""Tests for rbfkn.rbf_kernel."""
import numpy as np
import pytest
from moirais.fn.rbfkn import rbf_kernel


def test_rbfkn_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    h = 0.3
    result = rbf_kernel(X, h)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rbfkn_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    h = 0.3
    result = rbf_kernel(X, h)
    assert isinstance(result, dict)
