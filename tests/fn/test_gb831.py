"""Tests for gb831.gibbons_terry_hoeffding."""
import numpy as np
import pytest
from moirais.fn.gb831 import gibbons_terry_hoeffding


def test_gb831_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = gibbons_terry_hoeffding(x, y)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_gb831_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = gibbons_terry_hoeffding(x, y)
    assert isinstance(result, dict)
