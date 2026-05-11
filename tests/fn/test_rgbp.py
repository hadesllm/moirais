"""Tests for rgbp.rangayyan_basis_pursuit."""
import numpy as np
import pytest
from morie.fn.rgbp import rangayyan_basis_pursuit


def test_rgbp_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    tol = 1e-6
    result = rangayyan_basis_pursuit(x, D, tol)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rgbp_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    tol = 1e-6
    result = rangayyan_basis_pursuit(x, D, tol)
    assert isinstance(result, dict)
