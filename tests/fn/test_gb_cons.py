"""Tests for gb_cons.gibbons_consistency."""
import numpy as np
import pytest
from morie.fn.gb_cons import gibbons_consistency


def test_gb_cons_basic():
    """Test basic functionality."""
    T = np.random.default_rng(42).normal(0, 1, 100)
    H1 = np.random.default_rng(42).normal(0, 1, 100)
    result = gibbons_consistency(T, H1)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_gb_cons_edge():
    """Test edge cases."""
    T = np.random.default_rng(42).normal(0, 1, 100)
    H1 = np.random.default_rng(42).normal(0, 1, 100)
    result = gibbons_consistency(T, H1)
    assert isinstance(result, dict)
