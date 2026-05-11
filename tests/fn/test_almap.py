"""Tests for almap.alammar_mean_average_precision."""
import numpy as np
import pytest
from morie.fn.almap import alammar_mean_average_precision


def test_almap_basic():
    """Test basic functionality."""
    rankings_by_query = np.random.default_rng(42).normal(0, 1, 100)
    relevant_by_query = np.random.default_rng(42).normal(0, 1, 100)
    result = alammar_mean_average_precision(rankings_by_query, relevant_by_query)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_almap_edge():
    """Test edge cases."""
    rankings_by_query = np.random.default_rng(42).normal(0, 1, 100)
    relevant_by_query = np.random.default_rng(42).normal(0, 1, 100)
    result = alammar_mean_average_precision(rankings_by_query, relevant_by_query)
    assert isinstance(result, dict)
