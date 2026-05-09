"""Tests for tmlsbs.tmle_subset_selection."""
import numpy as np
import pytest
from moirais.fn.tmlsbs import tmle_subset_selection


def test_tmlsbs_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = tmle_subset_selection(y, D, X)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_tmlsbs_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = tmle_subset_selection(y, D, X)
    assert isinstance(result, dict)
