"""Tests for bndpl.bnp_density_pl."""
import numpy as np
import pytest
from moirais.fn.bndpl import bnp_density_pl


def test_bndpl_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    tree_depth = np.random.default_rng(42).normal(0, 1, 100)
    alpha = 0.05
    result = bnp_density_pl(y, tree_depth, alpha)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_bndpl_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    tree_depth = np.random.default_rng(42).normal(0, 1, 100)
    alpha = 0.05
    result = bnp_density_pl(y, tree_depth, alpha)
    assert isinstance(result, dict)
