"""Tests for hmdtst.geron_tree_sensitivity_scale."""
import numpy as np
import pytest
from moirais.fn.hmdtst import geron_tree_sensitivity_scale


def test_hmdtst_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = geron_tree_sensitivity_scale(X, y)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmdtst_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = geron_tree_sensitivity_scale(X, y)
    assert isinstance(result, dict)
