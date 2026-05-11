"""Tests for ghs022.ghosal_ch3_tailfree_cell_counts."""
import numpy as np
import pytest
from morie.fn.ghs022 import ghosal_ch3_tailfree_cell_counts


def test_ghs022_basic():
    """Test basic functionality."""
    X_i = np.random.default_rng(42).normal(0, 1, 100)
    A_epsilon = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    result = ghosal_ch3_tailfree_cell_counts(X_i, A_epsilon, n)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_ghs022_edge():
    """Test edge cases."""
    X_i = np.random.default_rng(42).normal(0, 1, 100)
    A_epsilon = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    result = ghosal_ch3_tailfree_cell_counts(X_i, A_epsilon, n)
    assert isinstance(result, dict)
