"""Tests for vanr1.vanraden_method1."""
import numpy as np
import pytest
from moirais.fn.vanr1 import vanraden_method1


def test_vanr1_basic():
    """Test basic functionality."""
    marker_matrix = np.random.default_rng(42).normal(0, 1, (10, 10))
    freq = np.random.default_rng(42).normal(0, 1, 100)
    result = vanraden_method1(marker_matrix, freq)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_vanr1_edge():
    """Test edge cases."""
    marker_matrix = np.random.default_rng(42).normal(0, 1, (10, 10))
    freq = np.random.default_rng(42).normal(0, 1, 100)
    result = vanraden_method1(marker_matrix, freq)
    assert isinstance(result, dict)
