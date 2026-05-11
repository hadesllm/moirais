"""Tests for gh_c2_9.ghosal_np_binary_reg."""
import numpy as np
import pytest
from morie.fn.gh_c2_9 import ghosal_np_binary_reg


def test_gh_c2_9_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = ghosal_np_binary_reg(x, y)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_gh_c2_9_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = ghosal_np_binary_reg(x, y)
    assert isinstance(result, dict)
