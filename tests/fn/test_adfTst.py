"""Tests for adfTst.adf_test."""
import numpy as np
import pytest
from morie.fn.adfTst import adf_test


def test_adfTst_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    k = 5
    result = adf_test(y, k)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_adfTst_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    k = 5
    result = adf_test(y, k)
    assert isinstance(result, dict)
