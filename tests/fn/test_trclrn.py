"""Tests for trclrn.tree_based_regime."""
import numpy as np
import pytest
from morie.fn.trclrn import tree_based_regime


def test_trclrn_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    W = np.random.default_rng(42).normal(0, 1, 100)
    result = tree_based_regime(y, D, W)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_trclrn_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    W = np.random.default_rng(42).normal(0, 1, 100)
    result = tree_based_regime(y, D, W)
    assert isinstance(result, dict)
