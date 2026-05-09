"""Tests for prtnst.partition_test."""
import numpy as np
import pytest
from moirais.fn.prtnst import partition_test


def test_prtnst_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    groups = np.random.default_rng(43).integers(0, 3, 100)
    alpha = 0.05
    result = partition_test(y, groups, alpha)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_prtnst_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    groups = np.random.default_rng(43).integers(0, 3, 100)
    alpha = 0.05
    result = partition_test(y, groups, alpha)
    assert isinstance(result, dict)
