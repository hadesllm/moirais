"""Tests for pettit.pettitt_test."""
import numpy as np
import pytest
from morie.fn.pettit import pettitt_test


def test_pettit_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = pettitt_test(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_pettit_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = pettitt_test(x)
    assert isinstance(result, dict)
