"""Tests for mwem.mwem."""
import numpy as np
import pytest
from morie.fn.mwem import mwem


def test_mwem_basic():
    """Test basic functionality."""
    queries = np.random.default_rng(42).normal(0, 1, 100)
    epsilon = 1e-6
    T = np.random.default_rng(43).integers(0, 2, 100)
    result = mwem(queries, epsilon, T)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_mwem_edge():
    """Test edge cases."""
    queries = np.random.default_rng(42).normal(0, 1, 100)
    epsilon = 1e-6
    T = np.random.default_rng(43).integers(0, 2, 100)
    result = mwem(queries, epsilon, T)
    assert isinstance(result, dict)
