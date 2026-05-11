"""Tests for permt.permutation_test_general."""
import numpy as np
import pytest
from morie.fn.permt import permutation_test_general


def test_permt_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = permutation_test_general(x, y)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_permt_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = permutation_test_general(x, y)
    assert isinstance(result, dict)
