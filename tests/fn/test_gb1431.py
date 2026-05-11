"""Tests for gb1431.gibbons_k2_contingency."""
import numpy as np
import pytest
from morie.fn.gb1431 import gibbons_k2_contingency


def test_gb1431_basic():
    """Test basic functionality."""
    table = np.random.default_rng(42).normal(0, 1, 100)
    result = gibbons_k2_contingency(table)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_gb1431_edge():
    """Test edge cases."""
    table = np.random.default_rng(42).normal(0, 1, 100)
    result = gibbons_k2_contingency(table)
    assert isinstance(result, dict)
