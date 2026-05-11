"""Tests for vanr2.vanraden_method2."""
import numpy as np
import pytest
from morie.fn.vanr2 import vanraden_method2


def test_vanr2_basic():
    """Test basic functionality."""
    marker_matrix = np.random.default_rng(42).normal(0, 1, (10, 10))
    weights = np.random.default_rng(45).exponential(1, 100)
    freq = np.random.default_rng(42).normal(0, 1, 100)
    result = vanraden_method2(marker_matrix, weights, freq)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_vanr2_edge():
    """Test edge cases."""
    marker_matrix = np.random.default_rng(42).normal(0, 1, (10, 10))
    weights = np.random.default_rng(45).exponential(1, 100)
    freq = np.random.default_rng(42).normal(0, 1, 100)
    result = vanraden_method2(marker_matrix, weights, freq)
    assert isinstance(result, dict)
