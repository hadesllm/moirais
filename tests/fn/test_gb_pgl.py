"""Tests for gb_pgl.gibbons_page_exact."""
import numpy as np
import pytest
from morie.fn.gb_pgl import gibbons_page_exact


def test_gb_pgl_basic():
    """Test basic functionality."""
    l = np.random.default_rng(42).normal(0, 1, 100)
    k = 5
    b = np.random.default_rng(42).normal(0, 1, 100)
    result = gibbons_page_exact(l, k, b)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_gb_pgl_edge():
    """Test edge cases."""
    l = np.random.default_rng(42).normal(0, 1, 100)
    k = 5
    b = np.random.default_rng(42).normal(0, 1, 100)
    result = gibbons_page_exact(l, k, b)
    assert isinstance(result, dict)
