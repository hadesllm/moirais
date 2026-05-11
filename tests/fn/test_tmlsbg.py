"""Tests for tmlsbg.tmle_subgroup."""
import numpy as np
import pytest
from morie.fn.tmlsbg import tmle_subgroup


def test_tmlsbg_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    subgroup = np.random.default_rng(42).normal(0, 1, 100)
    result = tmle_subgroup(y, D, X, subgroup)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_tmlsbg_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    subgroup = np.random.default_rng(42).normal(0, 1, 100)
    result = tmle_subgroup(y, D, X, subgroup)
    assert isinstance(result, dict)
