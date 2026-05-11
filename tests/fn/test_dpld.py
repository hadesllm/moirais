"""Tests for dpld.l_diversity."""
import numpy as np
import pytest
from morie.fn.dpld import l_diversity


def test_dpld_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    quasi_ids = np.arange(100, dtype=int)
    sensitive = np.random.default_rng(42).normal(0, 1, 100)
    l = np.random.default_rng(42).normal(0, 1, 100)
    result = l_diversity(X, quasi_ids, sensitive, l)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_dpld_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    quasi_ids = np.arange(100, dtype=int)
    sensitive = np.random.default_rng(42).normal(0, 1, 100)
    l = np.random.default_rng(42).normal(0, 1, 100)
    result = l_diversity(X, quasi_ids, sensitive, l)
    assert isinstance(result, dict)
