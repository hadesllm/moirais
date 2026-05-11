"""Tests for drspa.dr_spatial_did."""
import numpy as np
import pytest
from morie.fn.drspa import dr_spatial_did


def test_drspa_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    W_neighbors = np.random.default_rng(42).normal(0, 1, 100)
    result = dr_spatial_did(y, D, X, W_neighbors)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_drspa_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    W_neighbors = np.random.default_rng(42).normal(0, 1, 100)
    result = dr_spatial_did(y, D, X, W_neighbors)
    assert isinstance(result, dict)
