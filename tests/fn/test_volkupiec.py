"""Tests for volkupiec.vol_kupiec_var_test."""
import numpy as np
import pytest
from morie.fn.volkupiec import vol_kupiec_var_test


def test_volkupiec_basic():
    """Test basic functionality."""
    hits = np.random.default_rng(42).normal(0, 1, 100)
    alpha = 0.05
    result = vol_kupiec_var_test(hits, alpha)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_volkupiec_edge():
    """Test edge cases."""
    hits = np.random.default_rng(42).normal(0, 1, 100)
    alpha = 0.05
    result = vol_kupiec_var_test(hits, alpha)
    assert isinstance(result, dict)
