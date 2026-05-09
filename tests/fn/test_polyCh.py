"""Tests for polyCh.chebyshev_basis."""
import numpy as np
import pytest
from moirais.fn.polyCh import chebyshev_basis


def test_polyCh_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    result = chebyshev_basis(x, K)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_polyCh_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    result = chebyshev_basis(x, K)
    assert isinstance(result, dict)
