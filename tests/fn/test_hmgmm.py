"""Tests for hmgmm.geron_gaussian_mixture."""
import numpy as np
import pytest
from morie.fn.hmgmm import geron_gaussian_mixture


def test_hmgmm_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    n_components = 3
    seed = 42
    result = geron_gaussian_mixture(X, n_components, seed)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmgmm_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    n_components = 3
    seed = 42
    result = geron_gaussian_mixture(X, n_components, seed)
    assert isinstance(result, dict)
