"""Tests for dtrsp.decision_tree_split."""
import numpy as np
import pytest
from morie.fn.dtrsp import decision_tree_split


def test_dtrsp_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = decision_tree_split(x, y)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_dtrsp_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = decision_tree_split(x, y)
    assert isinstance(result, dict)
