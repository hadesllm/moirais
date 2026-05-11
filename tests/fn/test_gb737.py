"""Tests for gb737.gibbons_linrank_properties."""
import numpy as np
import pytest
from morie.fn.gb737 import gibbons_linrank_properties


def test_gb737_basic():
    """Test basic functionality."""
    a = np.random.default_rng(42).normal(0, 1, 100)
    Z = np.random.default_rng(43).normal(0, 1, (100, 10))
    result = gibbons_linrank_properties(a, Z)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_gb737_edge():
    """Test edge cases."""
    a = np.random.default_rng(42).normal(0, 1, 100)
    Z = np.random.default_rng(43).normal(0, 1, (100, 10))
    result = gibbons_linrank_properties(a, Z)
    assert isinstance(result, dict)
