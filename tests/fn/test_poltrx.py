"""Tests for poltrx.polya_tree_extended."""
import numpy as np
import pytest
from morie.fn.poltrx import polya_tree_extended


def test_poltrx_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    M = np.random.default_rng(43).normal(0, 1, (10, 10))
    alpha = 0.05
    result = polya_tree_extended(y, M, alpha)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_poltrx_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    M = np.random.default_rng(43).normal(0, 1, (10, 10))
    alpha = 0.05
    result = polya_tree_extended(y, M, alpha)
    assert isinstance(result, dict)
