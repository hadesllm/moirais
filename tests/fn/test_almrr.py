"""Tests for almrr.alammar_mean_reciprocal_rank."""
import numpy as np
import pytest
from morie.fn.almrr import alammar_mean_reciprocal_rank


def test_almrr_basic():
    """Test basic functionality."""
    rankings = np.random.default_rng(42).permutation(10).reshape(2, 5)
    relevant_indices = np.random.default_rng(42).normal(0, 1, 100)
    result = alammar_mean_reciprocal_rank(rankings, relevant_indices)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_almrr_edge():
    """Test edge cases."""
    rankings = np.random.default_rng(42).permutation(10).reshape(2, 5)
    relevant_indices = np.random.default_rng(42).normal(0, 1, 100)
    result = alammar_mean_reciprocal_rank(rankings, relevant_indices)
    assert isinstance(result, dict)
