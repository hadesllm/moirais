"""Tests for laplmo.laplacian_eigen."""
import numpy as np
import pytest
from morie.fn.laplmo import laplacian_eigen


def test_laplmo_basic():
    """Test basic functionality."""
    G = np.eye(10)
    result = laplacian_eigen(G)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_laplmo_edge():
    """Test edge cases."""
    G = np.eye(10)
    result = laplacian_eigen(G)
    assert isinstance(result, dict)
