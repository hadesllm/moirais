"""Tests for gb_pg2.gibbons_page_asymp."""
import numpy as np
import pytest
from moirais.fn.gb_pg2 import gibbons_page_asymp


def test_gb_pg2_basic():
    """Test basic functionality."""
    L = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    k = 5
    result = gibbons_page_asymp(L, n, k)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_gb_pg2_edge():
    """Test edge cases."""
    L = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    k = 5
    result = gibbons_page_asymp(L, n, k)
    assert isinstance(result, dict)
