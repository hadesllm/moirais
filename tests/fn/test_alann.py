"""Tests for alann.alammar_approximate_nearest_neighbor."""
import numpy as np
import pytest
from morie.fn.alann import alammar_approximate_nearest_neighbor


def test_alann_basic():
    """Test basic functionality."""
    query_vec = np.random.default_rng(42).normal(0, 1, 100)
    index = np.random.default_rng(42).normal(0, 1, 100)
    ef_search = np.random.default_rng(42).normal(0, 1, 100)
    result = alammar_approximate_nearest_neighbor(query_vec, index, ef_search)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_alann_edge():
    """Test edge cases."""
    query_vec = np.random.default_rng(42).normal(0, 1, 100)
    index = np.random.default_rng(42).normal(0, 1, 100)
    ef_search = np.random.default_rng(42).normal(0, 1, 100)
    result = alammar_approximate_nearest_neighbor(query_vec, index, ef_search)
    assert isinstance(result, dict)
