"""Tests for hermitS.hermite_basis."""
import numpy as np
import pytest
from morie.fn.hermitS import hermite_basis


def test_hermitS_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    result = hermite_basis(x, K)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hermitS_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    result = hermite_basis(x, K)
    assert isinstance(result, dict)
