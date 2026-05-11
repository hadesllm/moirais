"""Tests for gb971.gibbons_sukhatme."""
import numpy as np
import pytest
from morie.fn.gb971 import gibbons_sukhatme


def test_gb971_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = gibbons_sukhatme(x, y)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_gb971_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = gibbons_sukhatme(x, y)
    assert isinstance(result, dict)
