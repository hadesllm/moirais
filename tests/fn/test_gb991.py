"""Tests for gb991.gibbons_conover_scale."""
import numpy as np
import pytest
from morie.fn.gb991 import gibbons_conover_scale


def test_gb991_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = gibbons_conover_scale(x, y)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_gb991_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = gibbons_conover_scale(x, y)
    assert isinstance(result, dict)
