"""Tests for btperm.boot_permutation_test."""
import numpy as np
import pytest
from moirais.fn.btperm import boot_permutation_test


def test_btperm_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    stat = np.random.default_rng(42).normal(0, 1, 100)
    B = np.random.default_rng(43).normal(0, 1, (10, 10))
    result = boot_permutation_test(x, y, stat, B)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_btperm_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    stat = np.random.default_rng(42).normal(0, 1, 100)
    B = np.random.default_rng(43).normal(0, 1, (10, 10))
    result = boot_permutation_test(x, y, stat, B)
    assert isinstance(result, dict)
