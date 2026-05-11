"""Tests for ldiff.l_diversity_check."""
import numpy as np
import pytest
from morie.fn.ldiff import l_diversity_check


def test_ldiff_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    quasi_ids = np.arange(100, dtype=int)
    sensitive = np.random.default_rng(42).normal(0, 1, 100)
    l = np.random.default_rng(42).normal(0, 1, 100)
    result = l_diversity_check(y, quasi_ids, sensitive, l)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ldiff_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    quasi_ids = np.arange(100, dtype=int)
    sensitive = np.random.default_rng(42).normal(0, 1, 100)
    l = np.random.default_rng(42).normal(0, 1, 100)
    result = l_diversity_check(y, quasi_ids, sensitive, l)
    assert isinstance(result, dict)
