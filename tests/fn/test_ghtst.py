"""Tests for ghtst.ghosal_np_testing."""
import numpy as np
import pytest
from moirais.fn.ghtst import ghosal_np_testing


def test_ghtst_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ghosal_np_testing(x)
    assert 'statistic' in result
    assert 'p_value' in result
    assert 0 <= result['p_value'] <= 1


def test_ghtst_edge():
    """Test edge cases."""
    result = ghosal_np_testing(np.array([1.0]))
    assert result['n'] == 1
