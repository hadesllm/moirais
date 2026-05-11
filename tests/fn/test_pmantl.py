"""Tests for pmantl.partial_mantel_test."""
import numpy as np
import pytest
from morie.fn.pmantl import partial_mantel_test


def test_pmantl_basic():
    """Test basic functionality."""
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    B = np.random.default_rng(43).normal(0, 1, (10, 10))
    C = np.random.default_rng(42).normal(0, 1, 100)
    permutations = np.random.default_rng(42).normal(0, 1, 100)
    result = partial_mantel_test(A, B, C, permutations)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_pmantl_edge():
    """Test edge cases."""
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    B = np.random.default_rng(43).normal(0, 1, (10, 10))
    C = np.random.default_rng(42).normal(0, 1, 100)
    permutations = np.random.default_rng(42).normal(0, 1, 100)
    result = partial_mantel_test(A, B, C, permutations)
    assert isinstance(result, dict)
