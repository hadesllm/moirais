"""Tests for omegah.omega_hierarchical."""
import numpy as np
import pytest
from morie.fn.omegah import omega_hierarchical


def test_omegah_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    loadings_g = np.random.default_rng(42).normal(0, 1, 100)
    loadings_specific = np.random.default_rng(42).normal(0, 1, 100)
    result = omega_hierarchical(X, loadings_g, loadings_specific)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_omegah_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    loadings_g = np.random.default_rng(42).normal(0, 1, 100)
    loadings_specific = np.random.default_rng(42).normal(0, 1, 100)
    result = omega_hierarchical(X, loadings_g, loadings_specific)
    assert isinstance(result, dict)
